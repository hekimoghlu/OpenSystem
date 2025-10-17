/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 15, 2023.
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

//
// Copyright 2020 The ANGLE Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.
//
// HistogramWriter:
//   Helper class for writing histogram-json-set-format files to JSON.

#ifndef ANGLE_TESTS_TEST_UTILS_HISTOGRAM_WRITER_H_
#define ANGLE_TESTS_TEST_UTILS_HISTOGRAM_WRITER_H_

#if !defined(ANGLE_HAS_HISTOGRAMS)
#    error "Requires ANGLE_HAS_HISTOGRAMS, see angle_maybe_has_histograms"
#endif  // !defined(ANGLE_HAS_HISTOGRAMS)

#include <map>
#include <memory>
#include <string>

// Include forward delcarations for rapidjson types.
#include <rapidjson/fwd.h>

namespace catapult
{
class HistogramBuilder;
}  // namespace catapult

namespace angle
{
class HistogramWriter
{
  public:
    HistogramWriter();
    ~HistogramWriter();

    void addSample(const std::string &measurement,
                   const std::string &story,
                   double value,
                   const std::string &units);

    void getAsJSON(rapidjson::Document *doc) const;

  private:
#if ANGLE_HAS_HISTOGRAMS
    std::map<std::string, std::unique_ptr<catapult::HistogramBuilder>> mHistograms;
#endif  // ANGLE_HAS_HISTOGRAMS
};

// Define a stub implementation when histograms are compiled out.
#if !ANGLE_HAS_HISTOGRAMS
inline HistogramWriter::HistogramWriter()  = default;
inline HistogramWriter::~HistogramWriter() = default;
inline void HistogramWriter::addSample(const std::string &measurement,
                                       const std::string &story,
                                       double value,
                                       const std::string &units)
{}
inline void HistogramWriter::getAsJSON(rapidjson::Document *doc) const {}
#endif  // !ANGLE_HAS_HISTOGRAMS

}  // namespace angle

#endif  // ANGLE_TESTS_TEST_UTILS_HISTOGRAM_WRITER_H_
