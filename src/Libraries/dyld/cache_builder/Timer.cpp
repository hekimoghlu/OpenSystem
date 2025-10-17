/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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
#include "Timer.h"
#include "BuilderConfig.h"

#include <mach/mach_time.h>

using namespace cache_builder;

//
// MARK: --- cache_builder::Timer::Scope methods ---
//

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunguarded-availability-new"

cache_builder::Timer::Scope::Scope(const BuilderConfig& config, std::string_view name)
    : config(config)
    , name(name)
    , log(config.timer.log)
    , signpost(config.timer.signpost)
{
    os_signpost_emit_with_type(this->log, OS_SIGNPOST_INTERVAL_BEGIN, this->signpost, "dyld", "%s", name.data());

    // Record the start time if -time-passes was used.  We'll print it later when we go out of scope
    if ( config.log.printTimers )
        this->startTimeNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

cache_builder::Timer::Scope::~Scope()
{
    os_signpost_emit_with_type(this->log, OS_SIGNPOST_INTERVAL_END, this->signpost, "dyld", "%s", name.data());

    // Also print to stdout if -time-passes was passed to the builder
    if ( this->config.log.printTimers ) {
        uint64_t endTimeNanos   = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        uint64_t timeMillis     = (endTimeNanos - startTimeNanos) / 1000000;
        this->config.log.log("%s = %lldms\n", name.data(), timeMillis);
    }
}

#pragma clang diagnostic pop

//
// MARK: --- cache_builder::Timer methods ---
//

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunguarded-availability-new"

cache_builder::Timer::Timer()
{
    this->log      = os_log_create("com.apple.dyld", "cache-builder");
    this->signpost = os_signpost_id_generate(this->log);
}

#pragma clang diagnostic pop

//
// MARK: --- cache_builder::Timer::Scope methods ---
//

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunguarded-availability-new"

cache_builder::Timer::AggregateTimer::Scope::Scope(AggregateTimer& timer, std::string_view name)
    : timer(timer)
    , name(name)
{
    this->startTimeNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

cache_builder::Timer::AggregateTimer::Scope::~Scope()
{
    uint64_t endTimeNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    this->timer.record(this->name, this->startTimeNanos, endTimeNanos);
}

#pragma clang diagnostic pop

//
// MARK: --- cache_builder::Timer::AggregateTimer methods ---
//

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunguarded-availability-new"

cache_builder::Timer::AggregateTimer::AggregateTimer(const BuilderConfig& config)
    : config(config)
    , mapLock(PTHREAD_MUTEX_INITIALIZER)
{
}

cache_builder::Timer::AggregateTimer::~AggregateTimer()
{
    if ( this->config.log.printTimers ) {
        for ( const auto& nameAndTime : this->timesNanos ) {
            uint64_t timeMillis = nameAndTime.second / 1000000;
            this->config.log.log("%s = %lldms\n", nameAndTime.first.data(), timeMillis);
        }
    }
}

void cache_builder::Timer::AggregateTimer::record(std::string_view name, uint64_t startTime, uint64_t endTime)
{
    pthread_mutex_lock(&this->mapLock);
    // Use the map to look up the index in to the vector
    auto itAndInserted = timeMap.insert({ name, timesNanos.size() });
    if ( itAndInserted.second ) {
        // We added the name, so add a new element to the vector
        timesNanos.emplace_back(name, 0);
    }
    timesNanos[itAndInserted.first->second].second += (endTime - startTime);
    pthread_mutex_unlock(&this->mapLock);
}

#pragma clang diagnostic pop

#pragma clang diagnostic pop

//
// MARK: --- cache_builder::Stats methods ---
//

cache_builder::Stats::Stats(const BuilderConfig& config)
    : config(config)
{
}

cache_builder::Stats::~Stats()
{
    for ( const std::string& str : this->stats )
        this->config.log.log("%s", str.data());
}

void cache_builder::Stats::add(const char* format, ...)
{
    char*   output_string;
    va_list list;
    va_start(list, format);
    vasprintf(&output_string, format, list);
    va_end(list);

    this->stats.push_back(output_string);

    free(output_string);
}
