/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 13, 2025.
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
#ifndef SDK_OBJC_FRAMEWORK_CLASSES_VIDEOTOOLBOX_HELPERS_H_
#define SDK_OBJC_FRAMEWORK_CLASSES_VIDEOTOOLBOX_HELPERS_H_

#include <CoreFoundation/CoreFoundation.h>
#include <VideoToolbox/VideoToolbox.h>
#include <string>

// Convenience function for creating a dictionary.
inline CFDictionaryRef CreateCFTypeDictionary(CFTypeRef* keys,
                                              CFTypeRef* values,
                                              size_t size) {
  return CFDictionaryCreate(kCFAllocatorDefault, keys, values, size,
                            &kCFTypeDictionaryKeyCallBacks,
                            &kCFTypeDictionaryValueCallBacks);
}

// Copies characters from a CFStringRef into a std::string.
std::string CFStringToString(const CFStringRef cf_string);

// Convenience function for setting a VT property.
void SetVTSessionProperty(VTCompressionSessionRef session, CFStringRef key, int32_t value);

// Convenience function for setting a VT property.
void SetVTSessionProperty(VTCompressionSessionRef session, CFStringRef key, uint32_t value);
void SetVTSessionProperty(VTCompressionSessionRef session, CFStringRef key, double value);

// Convenience function for setting a VT property.
void SetVTSessionProperty(VTCompressionSessionRef session, CFStringRef key, bool value);

// Convenience function for setting a VT property.
void SetVTSessionProperty(VTCompressionSessionRef session, CFStringRef key, CFStringRef value);

// Convenience function for setting a VT property.
void SetVTSessionProperty(VTCompressionSessionRef session, CFStringRef key, CFArrayRef value);

#endif  // SDK_OBJC_FRAMEWORK_CLASSES_VIDEOTOOLBOX_HELPERS_H_
