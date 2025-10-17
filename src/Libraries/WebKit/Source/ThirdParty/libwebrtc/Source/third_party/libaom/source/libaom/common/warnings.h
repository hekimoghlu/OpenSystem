/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 27, 2025.
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
#ifndef AOM_COMMON_WARNINGS_H_
#define AOM_COMMON_WARNINGS_H_

#ifdef __cplusplus
extern "C" {
#endif

struct aom_codec_enc_cfg;
struct AvxEncoderConfig;

/*
 * Checks config for improperly used settings. Warns user upon encountering
 * settings that will lead to poor output quality. Prompts user to continue
 * when warnings are issued.
 */
void check_encoder_config(int disable_prompt,
                          const struct AvxEncoderConfig *global_config,
                          const struct aom_codec_enc_cfg *stream_config);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AOM_COMMON_WARNINGS_H_
