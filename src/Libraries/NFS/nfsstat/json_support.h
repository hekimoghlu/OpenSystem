/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 1, 2025.
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
#ifndef json_support_h
#define json_support_h

#include <CoreFoundation/CoreFoundation.h>

#pragma mark -

/*
 * Add object to an existing dictionary functions
 */
int json_dict_add_dict(CFMutableDictionaryRef dict, const char *key,
    const CFMutableDictionaryRef value);
int json_dict_add_array(CFMutableDictionaryRef dict, const char *key,
    const CFMutableArrayRef value);
int json_dict_add_num(CFMutableDictionaryRef dict, const char *key,
    const void *value, size_t size);
int json_dict_add_str(CFMutableDictionaryRef dict, const char *key,
    const char *value);

/*
 * Add object to an existing array functions
 */
int json_arr_add_str(CFMutableArrayRef arr,
    const char *value);
int json_arr_add_dict(CFMutableArrayRef arr,
    const CFMutableDictionaryRef value);

#pragma mark -

/*
 * Print out a Core Foundation object in JSON format
 */

int json_print_cf_object(CFTypeRef cf_object, char *output_file_path);

#endif /* json_support_h */
