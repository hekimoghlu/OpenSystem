/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 13, 2021.
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

#include <stdint.h>

void *language_retain_n(void *, uint32_t);
void language_release_n(void *, uint32_t);
void *language_nonatomic_retain_n(void *, uint32_t);
void language_nonatomic_release_n(void *, uint32_t);

void *language_unownedRetain_n(void *, uint32_t);
void language_unownedRelease_n(void *, uint32_t);
void *language_nonatomic_unownedRetain_n(void *, uint32_t);
void language_nonatomic_unownedRelease_n(void *, uint32_t);

// Wrappers so we can call these from Codira without upsetting the ARC optimizer.
void *wrapper_language_retain_n(void *obj, uint32_t n) {
  return language_retain_n(obj, n);
}

void wrapper_language_release_n(void *obj, uint32_t n) {
  language_release_n(obj, n);
}

void *wrapper_language_nonatomic_retain_n(void *obj, uint32_t n) {
  return language_nonatomic_retain_n(obj, n);
}

void wrapper_language_nonatomic_release_n(void *obj, uint32_t n) {
  language_nonatomic_release_n(obj, n);
}

void *wrapper_language_unownedRetain_n(void *obj, uint32_t n) {
  return language_unownedRetain_n(obj, n);
}

void wrapper_language_unownedRelease_n(void *obj, uint32_t n) {
  language_unownedRelease_n(obj, n);
}

void *wrapper_language_nonatomic_unownedRetain_n(void *obj, uint32_t n) {
  return language_nonatomic_unownedRetain_n(obj, n);
}

void wrapper_language_nonatomic_unownedRelease_n(void *obj, uint32_t n) {
  language_nonatomic_unownedRelease_n(obj, n);
}


