/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 12, 2025.
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
#include <xpc/xpc.h>
#include <CoreFoundation/CFError.h>
#include <CoreFoundation/CoreFoundation.h>

#ifndef _UTILITIES_SECURITYDXPC_H_
#define _UTILITIES_SECURITYDXPC_H_

bool SecXPCDictionarySetData(xpc_object_t message, const char *key, CFDataRef data, CFErrorRef *error);
bool SecXPCDictionarySetDataOptional(xpc_object_t message, const char *key, CFDataRef data, CFErrorRef *error);

bool SecXPCDictionarySetBool(xpc_object_t message, const char *key, bool value, CFErrorRef *error);

bool SecXPCDictionarySetPList(xpc_object_t message, const char *key, CFTypeRef object, CFErrorRef *error);
bool SecXPCDictionarySetPListWithRepair(xpc_object_t message, const char *key, CFTypeRef object, bool repair, CFErrorRef *error);
bool SecXPCDictionarySetPListOptional(xpc_object_t message, const char *key, CFTypeRef object, CFErrorRef *error);

bool SecXPCDictionarySetString(xpc_object_t message, const char *key, CFStringRef string, CFErrorRef *error);
bool SecXPCDictionarySetStringOptional(xpc_object_t message, const char *key, CFStringRef string, CFErrorRef *error);

bool SecXPCDictionarySetInt64(xpc_object_t message, const char *key, int64_t value, CFErrorRef *error);
bool SecXPCDictionarySetUInt64(xpc_object_t message, const char *key, uint64_t value, CFErrorRef *error);

bool SecXPCDictionarySetFileDescriptor(xpc_object_t message, const char *key, int fd, CFErrorRef *error);
int SecXPCDictionaryDupFileDescriptor(xpc_object_t message, const char *key, CFErrorRef *error);

CFTypeRef SecXPCDictionaryCopyPList(xpc_object_t message, const char *key, CFErrorRef *error);
bool SecXPCDictionaryCopyPListOptional(xpc_object_t message, const char *key, CFTypeRef *pobject, CFErrorRef *error);

CFArrayRef SecXPCDictionaryCopyArray(xpc_object_t message, const char *key, CFErrorRef *error);
bool SecXPCDictionaryCopyArrayOptional(xpc_object_t message, const char *key, CFArrayRef *parray, CFErrorRef *error);

CFDataRef SecXPCDictionaryCopyData(xpc_object_t message, const char *key, CFErrorRef *error);
bool SecXPCDictionaryCopyDataOptional(xpc_object_t message, const char *key, CFDataRef *pdata, CFErrorRef *error);

bool SecXPCDictionaryCopyURLOptional(xpc_object_t message, const char *key, CFURLRef *purl, CFErrorRef *error);

bool SecXPCDictionaryGetBool(xpc_object_t message, const char *key, CFErrorRef *error);

bool SecXPCDictionaryGetDouble(xpc_object_t message, const char *key, double *pvalue, CFErrorRef *error) ;

CFDictionaryRef SecXPCDictionaryCopyDictionary(xpc_object_t message, const char *key, CFErrorRef *error);
CFDictionaryRef SecXPCDictionaryCopyDictionaryWithoutZeroingDataInMessage(xpc_object_t message, const char *key, CFErrorRef *error);

bool SecXPCDictionaryCopyDictionaryOptional(xpc_object_t message, const char *key, CFDictionaryRef *pdictionary, CFErrorRef *error);

CFStringRef SecXPCDictionaryCopyString(xpc_object_t message, const char *key, CFErrorRef *error);

bool SecXPCDictionaryCopyStringOptional(xpc_object_t message, const char *key, CFStringRef *pstring, CFErrorRef *error);

CFSetRef SecXPCDictionaryCopySet(xpc_object_t message, const char *key, CFErrorRef *error);

/* For an xpc_array of datas */
bool SecXPCDictionaryCopyCFDataArrayOptional(xpc_object_t message, const char *key, CFArrayRef *data_array, CFErrorRef *error);

#endif
