/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 4, 2024.
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
#ifndef UTIL_H
# define UTIL_H

# include <sys/types.h>

# include <CoreFoundation/CoreFoundation.h>

extern int parseProperty(const char *, CFStringRef*, CFTypeRef*);
extern CFDictionaryRef ReadMetadata(const char*);
extern int WriteMetadata(const char *, CFDictionaryRef);
//extern uint32_t ChecksumData(CFDictionaryRef);
extern uint32_t ChecksumMetadata(const char *);
extern int UpdateChecksum(const char *, uint32_t);
extern int VerifyChecksum(const char *);
extern uint32_t GetChecksum(const char*);
extern uint32_t GetMetadataSize(const char*);
extern int IsAppleLabel(const char*);
extern uint32_t GetBlockSize(const char*);
extern uint64_t GetDiskSize(const char*);
extern int InitialMetadata(const char *, CFDictionaryRef, uint64_t);

extern int gDebug, gVerbose;

#endif /* UTIL_H */
