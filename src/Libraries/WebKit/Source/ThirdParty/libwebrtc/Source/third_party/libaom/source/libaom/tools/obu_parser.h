/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 29, 2024.
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
#ifndef AOM_TOOLS_OBU_PARSER_H_
#define AOM_TOOLS_OBU_PARSER_H_

#include <cstdint>

namespace aom_tools {

// Print information obtained from OBU(s) in data until data is exhausted or an
// error occurs. Returns true when all data is consumed successfully, and
// optionally reports OBU storage overhead via obu_overhead_bytes when the
// pointer is non-null.
bool DumpObu(const uint8_t *data, int length, int *obu_overhead_bytes);

}  // namespace aom_tools

#endif  // AOM_TOOLS_OBU_PARSER_H_
