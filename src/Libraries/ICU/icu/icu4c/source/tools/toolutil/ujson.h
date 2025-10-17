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

// Â© 2024 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html
#ifndef __UJSON_H__
#define __UJSON_H__

/*
  Without this code, the output if the JSON library code
  throws an exception would look like:
  terminate called after throwing an instance of 'nlohmann::json_abi_v3_11_3::detail::parse_error'
  what():  [json.exception.parse_error.101] parse error at line 1, column 1: attempting to parse an empty input; check that your input string or stream contains the expected JSON
Aborted (core dumped)

  (for example, if one of the JSON tests files contains an error or a file doesn't exist.)

  With this code, the output is:

  JSON exception thrown; modify tools/ctestfw//ujson.h to get diagnostics.
  Exiting immediately.

  The entire #if block can be commented out in order to temporarily enable exceptions
  and get a better parse error message (temporarily, while debugging).
 */

// Disable exceptions in JSON parser

#if _HAS_EXCEPTIONS == 0
#define JSON_TRY_USER if(true)
#define JSON_CATCH_USER(exception) if(false)
#define JSON_THROW_USER(exception) { \
    printf("JSON exception thrown; modify tools/toolutil/ujson.h to get diagnostics.\n\
Exiting immediately.\n"); \
    exit(1); \
}
#endif

#include "json-json.hpp"

#endif /* __UJSON_H__ */
