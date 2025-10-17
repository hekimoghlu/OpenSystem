/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 30, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef Timer_hpp
#define Timer_hpp

#include "BuilderOptions.h"

#include <os/signpost.h>
#include <string_view>

namespace cache_builder
{

struct BuilderConfig;

struct Timer
{
    Timer();

    struct Scope
    {
        Scope(const BuilderConfig& config, std::string_view name);
        ~Scope();

    private:
        const BuilderConfig&    config;
        std::string_view        name;
        os_log_t                log;
        os_signpost_id_t        signpost;
        uint64_t                startTimeNanos = 0;
    };

    // Similar to Scope, but aggregates multiple clients to a single time.  Eg, every dylib
    // has a "binding" Timer::Scope, and its too noisy to print them all for 2000 dylibs
    struct AggregateTimer
    {
        AggregateTimer(const BuilderConfig& config);
        ~AggregateTimer();
        void record(std::string_view name, uint64_t startTime, uint64_t endTime);

        // FIXME: Should we just have an AggregateTimer* in Timer::Scope instead?
        struct Scope
        {
            Scope(AggregateTimer& timer, std::string_view name);
            ~Scope();

        private:
            AggregateTimer&         timer;
            std::string_view        name;
            uint64_t                startTimeNanos = 0;
        };

    private:
        const BuilderConfig&                                config;
        std::unordered_map<std::string_view, uint32_t>      timeMap;
        std::vector<std::pair<std::string_view, uint64_t>>  timesNanos;
        pthread_mutex_t                                     mapLock;
    };

private:
    os_log_t            log;
    os_signpost_id_t    signpost;
};

struct Stats
{
    Stats(const BuilderConfig& config);
    ~Stats();

    void add(const char* format, ...)  __attribute__((format(printf, 2, 3)));

private:
    const BuilderConfig& config;
    std::vector<std::string> stats;
};

struct Logger
{
    Logger(const BuilderOptions& options);

    void log(const char* format, ...)  const __attribute__((format(printf, 2, 3)));

    bool printTimers    = false;
    bool printStats     = false;
    bool printDebug     = false;

private:
    std::string logPrefix;
};

} // namespace cache_builder

#endif /* Timer_hpp */
