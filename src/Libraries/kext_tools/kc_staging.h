/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 24, 2022.
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

//
//  kc_staging.h
//  kext_tools
//
//  Created by Jack Kim-Biggs on 7/16/19.
//

#ifndef kc_staging_h
#define kc_staging_h

#define _kOSKextReadOnlyDataVolumePath "/System/Volumes/Data"

#ifdef KCDITTO_STANDALONE_BINARY
#define ERROR_LOG_FUNCTION(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#define LOG_FUNCTION(fmt, ...) printf(fmt "\n", ##__VA_ARGS__)

#else /* KCDITTO_STANDALONE_BINARY */

#define ERROR_LOG_FUNCTION(fmt, ...) OSKextLog(NULL, \
		kOSKextLogFileAccessFlag | kOSKextLogErrorLevel, \
		fmt, ##__VA_ARGS__)
#define LOG_FUNCTION(fmt, ...) OSKextLog(NULL, \
		kOSKextLogFileAccessFlag | kOSKextLogBasicLevel, \
		fmt, ##__VA_ARGS__)
#endif /* !KCDITTO_STANDALONE_BINARY */

#define PASTE(x) #x
#define STRINGIFY(x) PASTE(x)
#define LOG_ERROR(...) do { \
	ERROR_LOG_FUNCTION(__FILE__ "." STRINGIFY(__LINE__) ": " __VA_ARGS__); \
} while (0)
#define LOG(...) LOG_FUNCTION(__VA_ARGS__)

int copyKCsInVolume(char *volRoot);
int copyDeferredPrelinkedKernels(char *volRoot);

#endif /* kc_staging_h */
