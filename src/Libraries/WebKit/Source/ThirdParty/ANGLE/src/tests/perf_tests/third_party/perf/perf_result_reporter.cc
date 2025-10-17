/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 30, 2022.
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

// Copyright 2019 The Chromium Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "perf_result_reporter.h"
#include "anglebase/logging.h"
#include "perf_test.h"

namespace perf_test
{

PerfResultReporter::PerfResultReporter(const std::string &metric_basename,
                                       const std::string &story_name)
    : metric_basename_(metric_basename), story_name_(story_name)
{}

PerfResultReporter::~PerfResultReporter() = default;

void PerfResultReporter::RegisterFyiMetric(const std::string &metric_suffix,
                                           const std::string &units)
{
    RegisterMetric(metric_suffix, units, false);
}

void PerfResultReporter::RegisterImportantMetric(const std::string &metric_suffix,
                                                 const std::string &units)
{
    RegisterMetric(metric_suffix, units, true);
}

void PerfResultReporter::AddResult(const std::string &metric_suffix, size_t value)
{
    auto iter = metric_map_.find(metric_suffix);
    CHECK(iter != metric_map_.end());

    PrintResult(metric_basename_, metric_suffix, story_name_, value, iter->second.units,
                iter->second.important);
}

void PerfResultReporter::AddResult(const std::string &metric_suffix, double value)
{
    auto iter = metric_map_.find(metric_suffix);
    CHECK(iter != metric_map_.end());

    PrintResult(metric_basename_, metric_suffix, story_name_, value, iter->second.units,
                iter->second.important);
}

void PerfResultReporter::AddResult(const std::string &metric_suffix, const std::string &value)
{
    auto iter = metric_map_.find(metric_suffix);
    CHECK(iter != metric_map_.end());

    PrintResult(metric_basename_, metric_suffix, story_name_, value, iter->second.units,
                iter->second.important);
}

bool PerfResultReporter::GetMetricInfo(const std::string &metric_suffix, MetricInfo *out)
{
    auto iter = metric_map_.find(metric_suffix);
    if (iter == metric_map_.end())
    {
        return false;
    }

    *out = iter->second;
    return true;
}

void PerfResultReporter::RegisterMetric(const std::string &metric_suffix,
                                        const std::string &units,
                                        bool important)
{
    CHECK(metric_map_.count(metric_suffix) == 0);
    metric_map_.insert({metric_suffix, {units, important}});
}

}  // namespace perf_test
