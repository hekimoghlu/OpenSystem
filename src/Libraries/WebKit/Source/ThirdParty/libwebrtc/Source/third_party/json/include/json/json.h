/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 30, 2022.
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
// Adapter for Google's jsoncpp project using json.hpp from nlohmann-json.
// See: <rdar://117694188> Add jsoncpp project to libwebrtc

#if WEBRTC_WEBKIT_BUILD

#define JSON_NOEXCEPTION
#include <nlohmann/v3.8/json.hpp>

namespace Json {

using String = nlohmann::json;
using Value = nlohmann::json;

class CharReader {
public:
    CharReader() = default;
    ~CharReader() = default;

    bool parse(char const* begin, char const* end, Value* root, String* /*error*/) {
        if (!root)
            return false;
        *root = nlohmann::json::parse(begin, end);
        return true;
    }
};

class CharReaderBuilder {
public:
    CharReaderBuilder() = default;
    ~CharReaderBuilder() = default;

    CharReader* newCharReader() { return new CharReader(); }
};

} // namespace Json

#endif // WEBRTC_WEBKIT_BUILD
