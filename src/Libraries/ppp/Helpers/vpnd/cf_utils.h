/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 22, 2025.
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
#ifndef __CF_UTILS_H__
#define __CF_UTILS_H__


Boolean isDictionary (CFTypeRef obj);
Boolean isArray (CFTypeRef obj);
Boolean isString (CFTypeRef obj);
Boolean isNumber (CFTypeRef obj);
Boolean isData (CFTypeRef obj);

int get_array_option(CFPropertyListRef options, CFStringRef entity, CFStringRef property, CFIndex index,
            u_char *opt, u_int32_t optsiz, u_int32_t *outlen, u_char *defaultval);

void get_str_option (CFPropertyListRef options, CFStringRef entity, CFStringRef property, 
                        u_char *opt, u_int32_t optsiz, u_int32_t *outlen, u_char *defaultval);

CFStringRef get_cfstr_option (CFPropertyListRef options, CFStringRef entity, CFStringRef property);

void get_int_option (CFPropertyListRef options, CFStringRef entity, CFStringRef property,
        u_int32_t *opt, u_int32_t defaultval);

Boolean GetIntFromDict (CFDictionaryRef dict, CFStringRef property, u_int32_t *outval, u_int32_t defaultval);

int GetStrFromDict (CFDictionaryRef dict, CFStringRef property, char *outstr, int maxlen, char *defaultval);

Boolean GetStrAddrFromDict (CFDictionaryRef dict, CFStringRef property, char *outstr, int maxlen);

Boolean GetStrNetFromDict (CFDictionaryRef dict, CFStringRef property, char *outstr, int maxlen);

#endif
