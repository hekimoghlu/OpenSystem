/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 14, 2022.
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
#ifndef __JSON_READER_H__
#define __JSON_READER_H__

#include "JSON.h"

class Diagnostics;

namespace json {

Node readJSON(Diagnostics& diags, const char* filePath, bool useJSON5);
Node readJSON(Diagnostics& diags, const void* contents, size_t length, bool useJSON5);

// Given a map node, returns the node representing the given value.
// If it is missing, returns a sentinel node and sets an error on the diagnostic
const Node& getRequiredValue(Diagnostics& diags, const Node& node, const char* key);

// Given a map node, returns the node representing the given value.
// If it is missing, return nullptr.
const Node* getOptionalValue(Diagnostics& diags, const Node& node, const char* key);

// Parses an int from the given node, or throws an error if its not an integer payload
uint64_t parseRequiredInt(Diagnostics& diags, const Node& node);

// Parses a bool from the given node, or throws an error if its not a boolean payload
bool parseRequiredBool(Diagnostics& diags, const Node& node);

// Parses a string from the given node, or throws an error if its not a string payload
const std::string& parseRequiredString(Diagnostics& diags, const Node& node);


} // namespace json


#endif // __JSON_READER_H__
