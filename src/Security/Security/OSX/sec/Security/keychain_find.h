/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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
#ifndef _KEYCHAIN_FINDINTERNETPASSWORD_H_
#define _KEYCHAIN_FINDINTERNETPASSWORD_H_  1

#ifdef __cplusplus
extern "C" {
#endif

extern int keychain_item(int argc, char * const *argv);

extern int keychain_find_internet_password(int argc, char * const *argv);

extern int keychain_find_generic_password(int argc, char * const *argv);

extern int keychain_find_certificate(int argc, char * const *argv);

extern int keychain_delete_internet_password(int argc, char * const *argv);

extern int keychain_delete_generic_password(int argc, char * const *argv);

extern int keychain_dump(int argc, char * const *argv);

extern int keychain_show_certificates(int argc, char * const *argv);

#ifdef __cplusplus
}
#endif

#endif /* _KEYCHAIN_FINDINTERNETPASSWORD_H_ */
