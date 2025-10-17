/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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
#pragma once

#include <string>
#include <vector>

extern const char* const kZipFileSeparator;

void format_string(std::string* str, const std::vector<std::pair<std::string, std::string>>& params);

bool file_is_in_dir(const std::string& file, const std::string& dir);
bool file_is_under_dir(const std::string& file, const std::string& dir);
bool normalize_path(const char* path, std::string* normalized_path);
bool parse_zip_path(const char* input_path, std::string* zip_path, std::string* entry_path);

// For every path element this function checks of it exists, and is a directory,
// and normalizes it:
// 1. For regular path it converts it to realpath()
// 2. For path in a zip file it uses realpath on the zipfile
//    normalizes entry name by calling normalize_path function.
void resolve_paths(std::vector<std::string>& paths,
                   std::vector<std::string>* resolved_paths);
// Resolve a single path. Return empty string when the path is invalid or can't
// be resolved.
std::string resolve_path(const std::string& path);

void split_path(const char* path, const char* delimiters, std::vector<std::string>* paths);

std::string dirname(const char* path);

bool safe_add(off64_t* out, off64_t a, size_t b);
bool is_first_stage_init();
