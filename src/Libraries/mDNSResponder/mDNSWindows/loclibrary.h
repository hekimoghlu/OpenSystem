/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 11, 2024.
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
/* loclibrary.h                                                      
 * ----------------------------------------------------------------------
 * Header file for localization library                                       
 * Originally created by jsantamaria: 3 may 2004                         
 * ----------------------------------------------------------------------
 */
 
#ifndef _loclibrary_h_
#define _loclibrary_h_


#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

int PathForResourceW ( HMODULE module, const wchar_t *name, wchar_t *locFile, int locFileLen);
int PathForResourceWithPathW ( const wchar_t *path, const wchar_t *name, wchar_t *locFile, int locFileLen);

int PathForResourceA ( HMODULE module, const char *name, char *locFile, int locFileLen);
int PathForResourceWithPathA ( const char *path, const char *name, char *locFile, int locFileLen);


#ifdef UNICODE
#define PathForResource PathForResourceW
#define PathForResourceWithPath PathForResourceWithPathW
#else
#define PathForResource PathForResourceA
#define PathForResourceWithPath PathForResourceWithPathA
#endif // UNICODE


#ifdef __cplusplus
}
#endif // __cplusplus


#endif // _loclibrary_h_
