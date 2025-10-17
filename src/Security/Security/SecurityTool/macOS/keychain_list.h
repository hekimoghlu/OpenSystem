/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 18, 2022.
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
#ifndef _KEYCHAIN_LIST_H_
#define _KEYCHAIN_LIST_H_  1

#ifdef __cplusplus
extern "C" {
#endif

extern int keychain_list(int argc, char * const *argv);
extern int ctk_list(int argc, char * const *argv);

extern int keychain_default(int argc, char * const *argv);

extern int keychain_login(int argc, char * const *argv);

#ifdef __cplusplus
}
#endif

#endif /* _KEYCHAIN_LIST_H_ */
