/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 9, 2022.
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
#include "helpers.h"

#include <string>

#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

// Copies characters from a CFStringRef into a std::string.
std::string CFStringToString(const CFStringRef cf_string) {
  RTC_DCHECK(cf_string);
  std::string std_string;
  // Get the size needed for UTF8 plus terminating character.
  size_t buffer_size =
      CFStringGetMaximumSizeForEncoding(CFStringGetLength(cf_string),
                                        kCFStringEncodingUTF8) +
      1;
  std::unique_ptr<char[]> buffer(new char[buffer_size]);
  if (CFStringGetCString(cf_string, buffer.get(), buffer_size,
                         kCFStringEncodingUTF8)) {
    // Copy over the characters.
    std_string.assign(buffer.get());
  }
  return std_string;
}

// Convenience function for setting a VT property.
void SetVTSessionProperty(VTCompressionSessionRef vtSession, CFStringRef key, int32_t value) {
  RTC_DCHECK(vtSession);
  CFNumberRef cfNum =
      CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &value);
  OSStatus status = VTSessionSetProperty(vtSession, key, cfNum);
  CFRelease(cfNum);
  if (status != noErr) {
    std::string key_string = CFStringToString(key);
    RTC_LOG(LS_ERROR) << "VTSessionSetProperty failed to set: " << key_string
                      << " to " << value << ": " << status;
  }
}

// Convenience function for setting a VT property.
void SetVTSessionProperty(VTCompressionSessionRef vtSession, CFStringRef key, double value) {
  RTC_DCHECK(vtSession);
  CFNumberRef cfNum =
      CFNumberCreate(kCFAllocatorDefault, kCFNumberDoubleType, &value);
  OSStatus status = VTSessionSetProperty(vtSession, key, cfNum);
  CFRelease(cfNum);
  if (status != noErr) {
    std::string key_string = CFStringToString(key);
    RTC_LOG(LS_ERROR) << "VTSessionSetProperty failed to set: " << key_string
                      << " to " << value << ": " << status;
  }
}

void SetVTSessionProperty(VTCompressionSessionRef vtSession, CFStringRef key, uint32_t value) {
  RTC_DCHECK(vtSession);
  int64_t value_64 = value;
  CFNumberRef cfNum =
      CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt64Type, &value_64);
  OSStatus status = VTSessionSetProperty(vtSession, key, cfNum);
  CFRelease(cfNum);
  if (status != noErr) {
    std::string key_string = CFStringToString(key);
    RTC_LOG(LS_ERROR) << "VTSessionSetProperty failed to set: " << key_string
                      << " to " << value << ": " << status;
  }
}

// Convenience function for setting a VT property.
void SetVTSessionProperty(VTCompressionSessionRef vtSession, CFStringRef key, bool value) {
  RTC_DCHECK(vtSession);
  CFBooleanRef cf_bool = (value) ? kCFBooleanTrue : kCFBooleanFalse;
  OSStatus status = VTSessionSetProperty(vtSession, key, cf_bool);
  if (status != noErr) {
    std::string key_string = CFStringToString(key);
    RTC_LOG(LS_ERROR) << "VTSessionSetProperty failed to set: " << key_string
                      << " to " << value << ": " << status;
  }
}

// Convenience function for setting a VT property.
void SetVTSessionProperty(VTCompressionSessionRef vtSession, CFStringRef key, CFStringRef value) {
  RTC_DCHECK(vtSession);
  OSStatus status = VTSessionSetProperty(vtSession, key, value);
  if (status != noErr) {
    std::string key_string = CFStringToString(key);
    std::string val_string = CFStringToString(value);
    RTC_LOG(LS_ERROR) << "VTSessionSetProperty failed to set: " << key_string
                      << " to " << val_string << ": " << status;
  }
}

// Convenience function for setting a VT property.
void SetVTSessionProperty(VTCompressionSessionRef vtSession, CFStringRef key, CFArrayRef value) {
  RTC_DCHECK(vtSession);
  OSStatus status = VTSessionSetProperty(vtSession, key, value);

  if (status != noErr) {
    std::string key_string = CFStringToString(key);
    RTC_LOG(LS_ERROR) << "VTSessionSetProperty failed to set: " << key_string
                      << " to array value: " << status;
  }
}
