//
//  llm_demo.cpp
//
//  Created by MNN on 2023/03/24.
//  ZhaodeWang
//

#include "llm/llm.hpp"
#define MNN_OPEN_TIME_TRACE
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <initializer_list>
//#define LLM_SUPPORT_AUDIO
#ifdef LLM_SUPPORT_AUDIO
#include "audio/audio.hpp"
#endif
using namespace MNN::Transformer;

static void append_metrics_jsonl(const std::string& path, int prompt_index, const LlmContext* context) {
    if (path.empty() || context == nullptr) {
        return;
    }
    std::ofstream of(path, std::ios::app);
    if (!of.good()) {
        return;
    }
    of << "{\"prompt_index\":" << prompt_index
       << ",\"prompt_tokens\":" << context->prompt_len
       << ",\"decode_tokens\":" << context->gen_seq_len
       << ",\"prefill_us\":" << context->prefill_us
       << ",\"decode_us\":" << context->decode_us
       << ",\"sample_us\":" << context->sample_us
       << ",\"h2o_keep_ratio\":" << context->h2o_keep_ratio
       << ",\"h2o_lossy_ratio\":" << context->h2o_lossy_ratio
       << ",\"h2o_lossless_ratio\":" << context->h2o_lossless_ratio
       << ",\"h2o_lossless_raw_bytes\":" << context->h2o_lossless_raw_bytes
       << ",\"h2o_lossless_compressed_bytes\":" << context->h2o_lossless_compressed_bytes
       << ",\"h2o_lossless_decompressed_bytes\":" << context->h2o_lossless_decompressed_bytes
       << ",\"h2o_lossless_compress_us\":" << context->h2o_lossless_compress_us
       << ",\"h2o_lossless_decompress_us\":" << context->h2o_lossless_decompress_us
       << ",\"h2o_lossless_queue_depth_peak\":" << context->h2o_lossless_queue_depth_peak
       << ",\"h2o_lossless_fallback_count\":" << context->h2o_lossless_fallback_count
       << ",\"h2o_lossless_backpressure_skip\":" << context->h2o_lossless_backpressure_skip_count
       << ",\"h2o_lossless_async_queue_peak\":" << context->h2o_lossless_async_queue_peak
       << ",\"h2o_lossless_async_wait_us\":" << context->h2o_lossless_async_wait_us
       << ",\"h2o_lossless_decode_cache_hit\":" << context->h2o_lossless_decode_cache_hit
       << ",\"h2o_lossless_decode_cache_miss\":" << context->h2o_lossless_decode_cache_miss
       << ",\"h2o_last_evict_tokens\":" << context->h2o_last_evict_tokens
       << ",\"h2o_total_evict_tokens\":" << context->h2o_total_evict_tokens
       << "}\n";
}

static void tuning_prepare(Llm* llm) {
    MNN_PRINT("Prepare for tuning opt Begin\n");
    llm->tuning(OP_ENCODER_NUMBER, {1, 5, 10, 20, 30, 50, 100});
    MNN_PRINT("Prepare for tuning opt End\n");
}

std::vector<std::vector<std::string>> parse_csv(const std::vector<std::string>& lines) {
    std::vector<std::vector<std::string>> csv_data;
    std::string line;
    std::vector<std::string> row;
    std::string cell;
    bool insideQuotes = false;
    bool startCollecting = false;

    // content to stream
    std::string content = "";
    for (auto line : lines) {
        content = content + line + "\n";
    }
    std::istringstream stream(content);

    while (stream.peek() != EOF) {
        char c = stream.get();
        if (c == '"') {
            if (insideQuotes && stream.peek() == '"') { // quote
                cell += '"';
                stream.get(); // skip quote
            } else {
                insideQuotes = !insideQuotes; // start or end text in quote
            }
            startCollecting = true;
        } else if (c == ',' && !insideQuotes) { // end element, start new element
            row.push_back(cell);
            cell.clear();
            startCollecting = false;
        } else if ((c == '\n' || stream.peek() == EOF) && !insideQuotes) { // end line
            row.push_back(cell);
            csv_data.push_back(row);
            cell.clear();
            row.clear();
            startCollecting = false;
        } else {
            cell += c;
            startCollecting = true;
        }
    }
    return csv_data;
}

static int benchmark(Llm* llm, const std::vector<std::string>& prompts, int max_token_number, const std::string& metrics_jsonl) {
    int prompt_len = 0;
    int decode_len = 0;
    int64_t prefill_time = 0;
    int64_t decode_time = 0;
    int64_t sample_time = 0;
    // llm->warmup();
    auto context = llm->getContext();
    if (max_token_number > 0) {
        // Respect caller-provided decode budget for benchmark/eval scripts.
        std::ostringstream cfg;
        cfg << "{\"max_new_tokens\":" << max_token_number << "}";
        llm->set_config(cfg.str());
    }
#ifdef LLM_SUPPORT_AUDIO
    std::vector<float> waveform;
    llm->setWavformCallback([&](const float* ptr, size_t size, bool last_chunk) {
        waveform.reserve(waveform.size() + size);
        waveform.insert(waveform.end(), ptr, ptr + size);
        if (last_chunk) {
            auto waveform_var = MNN::Express::_Const(waveform.data(), {(int)waveform.size()}, MNN::Express::NCHW, halide_type_of<float>());
            MNN::AUDIO::save("output.wav", waveform_var, 24000);
            waveform.clear();
        }
        return true;
    });
#endif
    for (int i = 0; i < prompts.size(); i++) {
        auto prompt = prompts[i];
     // #define MIMO_NO_THINKING
     #ifdef MIMO_NO_THINKING
        // update config.json and llm_config.json if need. example:
        llm->set_config("{\"assistant_prompt_template\":\"<|im_start|>assistant\\n<think>\\n</think>\%s<|im_end|>\\n\"}");
        prompt = prompt + "<think>\n</think>";
     #endif

        // prompt start with '#' will be ignored
        if (prompt.substr(0, 1) == "#") {
            continue;
        }
        
        if (max_token_number >= 0) {
            llm->response(prompt, &std::cout, nullptr, max_token_number);
        } else {
            llm->response(prompt);
        }
        prompt_len += context->prompt_len;
        decode_len += context->gen_seq_len;
        prefill_time += context->prefill_us;
        decode_time += context->decode_us;
        sample_time += context->sample_us;
        append_metrics_jsonl(metrics_jsonl, i, context);
    }
    llm->generateWavform();

    float vision_s = context->vision_us / 1e6;
    float audio_s = context->audio_us / 1e6;
    float prefill_s = prefill_time / 1e6;
    float decode_s = decode_time / 1e6;
    float sample_s = sample_time / 1e6;
    float vision_speed = 0.0f;
    if (context->pixels_mp > 0.0f) {
        vision_speed = context->pixels_mp / vision_s;
    }
    float audio_speed = 0.0f;
    if (context->audio_input_s > 0.0f) {
        audio_speed = context->audio_input_s / audio_s;
    }
    MNN_PRINT("\n#################################\n");
    MNN_PRINT("prompt tokens num = %d\n", prompt_len);
    MNN_PRINT("decode tokens num = %d\n", decode_len);
    MNN_PRINT(" vision time = %.2f s\n", vision_s);
    MNN_PRINT(" pixels_mp = %.2f MP\n", context->pixels_mp);
    MNN_PRINT("  audio process time = %.2f s\n", audio_s);
    MNN_PRINT("  audio input time = %.2f s\n", context->audio_input_s);
    MNN_PRINT("prefill time = %.2f s\n", prefill_s);
    MNN_PRINT(" decode time = %.2f s\n", decode_s);
    MNN_PRINT(" sample time = %.2f s\n", sample_s);
    MNN_PRINT("prefill speed = %.2f tok/s\n", prompt_len / prefill_s);
    MNN_PRINT(" decode speed = %.2f tok/s\n", decode_len / decode_s);
    MNN_PRINT(" vision speed = %.3f MP/s\n", vision_speed);
    MNN_PRINT(" audio RTF = %.3f \n", audio_s / context->audio_input_s);
    MNN_PRINT("##################################\n");
    return 0;
}

static int ceval(Llm* llm, const std::vector<std::string>& lines, std::string filename) {
    auto csv_data = parse_csv(lines);
    int right = 0, wrong = 0;
    std::vector<std::string> answers;
    for (int i = 1; i < csv_data.size(); i++) {
        const auto& elements = csv_data[i];
        std::string prompt = elements[1];
        prompt += "\n\nA. " + elements[2];
        prompt += "\nB. " + elements[3];
        prompt += "\nC. " + elements[4];
        prompt += "\nD. " + elements[5];
        prompt += "\n\n";
        MNN_PRINT("%s", prompt.c_str());
        MNN_PRINT("## 进度: %d / %lu\n", i, lines.size() - 1);
        std::ostringstream lineOs;
        llm->response(prompt.c_str(), &lineOs);
        auto line = lineOs.str();
        MNN_PRINT("%s", line.c_str());
        answers.push_back(line);
    }
    {
        auto position = filename.rfind("/");
        if (position != std::string::npos) {
            filename = filename.substr(position + 1, -1);
        }
        position = filename.find("_val");
        if (position != std::string::npos) {
            filename.replace(position, 4, "_res");
        }
        std::cout << "store to " << filename << std::endl;
    }
    std::ofstream ofp(filename);
    ofp << "id,answer" << std::endl;
    for (int i = 0; i < answers.size(); i++) {
        auto& answer = answers[i];
        ofp << i << ",\""<< answer << "\"" << std::endl;
    }
    ofp.close();
    return 0;
}

static int eval(Llm* llm, std::string prompt_file, int max_token_number, const std::string& metrics_jsonl, bool prompt_file_as_whole) {
    std::cout << "prompt file is " << prompt_file << std::endl;
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;
    std::string prompt;
    if (prompt_file_as_whole) {
        std::ostringstream tempOs;
        tempOs << prompt_fs.rdbuf();
        prompt = tempOs.str();
        while (!prompt.empty() && (prompt.back() == '\n' || prompt.back() == '\r')) {
            prompt.pop_back();
        }
        if (!prompt.empty()) {
            prompts = {prompt};
        }
    } else {
        while (std::getline(prompt_fs, prompt)) {
            if (prompt.empty()) {
                continue;
            }
            if (prompt.back() == '\r') {
                prompt.pop_back();
            }
            prompts.push_back(prompt);
        }
    }
    prompt_fs.close();
    if (prompts.empty()) {
        return 1;
    }
    // ceval
    if (prompts[0] == "id,question,A,B,C,D,answer") {
        return ceval(llm, prompts, prompt_file);
    }
    return benchmark(llm, prompts, max_token_number, metrics_jsonl);
}

void chat(Llm* llm) {
    ChatMessages messages;
    messages.emplace_back("system", "You are a helpful assistant.");
    auto context = llm->getContext();
    while (true) {
        std::cout << "\nUser: ";
        std::string user_str;
        std::getline(std::cin, user_str);
        if (user_str == "/exit") {
            return;
        }
        if (user_str == "/reset") {
            llm->reset();
            std::cout << "\nA: reset done." << std::endl;
            continue;
        }
        messages.emplace_back("user", user_str);
        std::cout << "\nA: " << std::flush;
        llm->response(messages);
        auto assistant_str = context->generate_str;
        messages.emplace_back("assistant", assistant_str);
    }
}
int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " config.json <prompt.txt>" << std::endl;
        return 1;
    }
    MNN::BackendConfig backendConfig;
    auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
    MNN::Express::ExecutorScope s(executor);

    std::string config_path = argv[1];
    std::cout << "config path is " << config_path << std::endl;
    std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
    std::string tmp_path = "tmp";
    if (const char* env_tmp_path = std::getenv("LLM_DEMO_TMP_PATH")) {
        if (env_tmp_path[0] != '\0') {
            tmp_path = env_tmp_path;
        }
    }
    llm->set_config(std::string("{\"tmp_path\":\"") + tmp_path + "\"}");
    {
        AUTOTIME;
        bool res = llm->load();
        if (!res) {
            MNN_ERROR("LLM init error\n");
            return 1;
        }
    }
    if (true) {
        AUTOTIME;
        tuning_prepare(llm.get());
        // Tuning may execute warmup decode/prefill passes that mutate KV/H2O runtime
        // state. Ensure real eval starts from a clean context.
        llm->reset();
    }
    if (argc < 3) {
        chat(llm.get());
        return 0;
    }
    int max_token_number = -1;
    if (argc >= 4) {
        std::istringstream os(argv[3]);
        os >> max_token_number;
    }
    std::string metrics_jsonl;
    if (const char* env_metrics = std::getenv("LLM_DEMO_METRICS_JSONL")) {
        metrics_jsonl = env_metrics;
    }
    bool disable_thinking = false;
    bool prompt_file_as_whole = false;
    for (int i = 4; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--metrics-jsonl=", 0) == 0) {
            metrics_jsonl = arg.substr(std::string("--metrics-jsonl=").size());
            continue;
        }
        if (arg == "--metrics-jsonl" && (i + 1) < argc) {
            metrics_jsonl = argv[++i];
            continue;
        }
        if (arg == "--no-thinking") {
            disable_thinking = true;
            continue;
        }
        if (arg == "--prompt-file-mode=whole" || arg == "--prompt-file-as-whole") {
            prompt_file_as_whole = true;
            continue;
        }
        if (arg == "--prompt-file-mode" && (i + 1) < argc) {
            std::string mode = argv[++i];
            if (mode == "whole" || mode == "file") {
                prompt_file_as_whole = true;
            }
            continue;
        }
        // Keep backward compatibility: any extra 5th positional arg toggles no-thinking.
        disable_thinking = true;
    }
    if (disable_thinking) {
        MNN_PRINT("Set not thinking, only valid for Qwen3\n");
        llm->set_config(R"({
            "jinja": {
                "context": {
                    "enable_thinking":false
                }
            }
        })");
    }
    std::string prompt_file = argv[2];
    llm->set_config(R"({
        "async":false
    })");
    return eval(llm.get(), prompt_file, max_token_number, metrics_jsonl, prompt_file_as_whole);
}
