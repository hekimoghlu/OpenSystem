/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 16, 2024.
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
#ifndef RTC_BASE_STRINGS_JSON_H_
#define RTC_BASE_STRINGS_JSON_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "json/json.h"

namespace rtc {

///////////////////////////////////////////////////////////////////////////////
// JSON Helpers
///////////////////////////////////////////////////////////////////////////////

// Robust conversion operators, better than the ones in JsonCpp.
bool GetIntFromJson(const Json::Value& in, int* out);
bool GetUIntFromJson(const Json::Value& in, unsigned int* out);
bool GetStringFromJson(const Json::Value& in, std::string* out);
bool GetBoolFromJson(const Json::Value& in, bool* out);
bool GetDoubleFromJson(const Json::Value& in, double* out);

// Pull values out of a JSON array.
bool GetValueFromJsonArray(const Json::Value& in, size_t n, Json::Value* out);
bool GetIntFromJsonArray(const Json::Value& in, size_t n, int* out);
bool GetUIntFromJsonArray(const Json::Value& in, size_t n, unsigned int* out);
bool GetStringFromJsonArray(const Json::Value& in, size_t n, std::string* out);
bool GetBoolFromJsonArray(const Json::Value& in, size_t n, bool* out);
bool GetDoubleFromJsonArray(const Json::Value& in, size_t n, double* out);

// Convert json arrays to std::vector
bool JsonArrayToValueVector(const Json::Value& in,
                            std::vector<Json::Value>* out);
bool JsonArrayToIntVector(const Json::Value& in, std::vector<int>* out);
bool JsonArrayToUIntVector(const Json::Value& in,
                           std::vector<unsigned int>* out);
bool JsonArrayToStringVector(const Json::Value& in,
                             std::vector<std::string>* out);
bool JsonArrayToBoolVector(const Json::Value& in, std::vector<bool>* out);
bool JsonArrayToDoubleVector(const Json::Value& in, std::vector<double>* out);

// Convert std::vector to json array
Json::Value ValueVectorToJsonArray(const std::vector<Json::Value>& in);
Json::Value IntVectorToJsonArray(const std::vector<int>& in);
Json::Value UIntVectorToJsonArray(const std::vector<unsigned int>& in);
Json::Value StringVectorToJsonArray(const std::vector<std::string>& in);
Json::Value BoolVectorToJsonArray(const std::vector<bool>& in);
Json::Value DoubleVectorToJsonArray(const std::vector<double>& in);

// Pull values out of a JSON object.
bool GetValueFromJsonObject(const Json::Value& in,
                            absl::string_view k,
                            Json::Value* out);
bool GetIntFromJsonObject(const Json::Value& in, absl::string_view k, int* out);
bool GetUIntFromJsonObject(const Json::Value& in,
                           absl::string_view k,
                           unsigned int* out);
bool GetStringFromJsonObject(const Json::Value& in,
                             absl::string_view k,
                             std::string* out);
bool GetBoolFromJsonObject(const Json::Value& in,
                           absl::string_view k,
                           bool* out);
bool GetDoubleFromJsonObject(const Json::Value& in,
                             absl::string_view k,
                             double* out);

// Writes out a Json value as a string.
std::string JsonValueToString(const Json::Value& json);

}  // namespace rtc

#endif  // RTC_BASE_STRINGS_JSON_H_
