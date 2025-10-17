/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 20, 2024.
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

#include <fuzzer/FuzzedDataProvider.h>
#include <string>

#include "jq.h"
#include "jv.h"

// Fuzzer inspired by /src/jq_test.c
// The goal is to have the fuzzer execute the functions:
// jq_compile -> jv_parse -> jq_next.
extern "C" int LLVMFuzzerTestOneInput(uint8_t *data, size_t size) {
  FuzzedDataProvider fdp(data, size);
  std::string prog_payload = fdp.ConsumeRandomLengthString();
  std::string parse_payload1 = fdp.ConsumeRandomLengthString();
  std::string parse_payload2 = fdp.ConsumeRandomLengthString();

  jq_state *jq = NULL;
  jq = jq_init();
  if (jq != NULL) {
    jq_set_attr(jq, jv_string("JQ_ORIGIN"), jv_string("/tmp/"));

    if (jq_compile(jq, prog_payload.c_str())) {
      // Process to jv_parse and then jv_next
      jv input = jv_parse(parse_payload1.c_str());
      if (jv_is_valid(input)) {
        jq_start(jq, input, 0);
        jv next = jv_parse(parse_payload2.c_str());
        if (jv_is_valid(next)) {
          jv actual = jq_next(jq);
          jv_free(actual);
        }
        jv_free(next);
      } else {
        // Only free if input is invalid as otherwise jq_teardown
        // frees it.
        jv_free(input);
      }
    }
  }
  jq_teardown(&jq);

  return 0;
}
