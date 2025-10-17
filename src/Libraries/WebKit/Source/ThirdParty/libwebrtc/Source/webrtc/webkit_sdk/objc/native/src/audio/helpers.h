/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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
#ifndef SDK_OBJC_NATIVE_SRC_AUDIO_HELPERS_H_
#define SDK_OBJC_NATIVE_SRC_AUDIO_HELPERS_H_

#include <string>

namespace webrtc {
namespace ios {

bool CheckAndLogError(BOOL success, NSError* error);

NSString* NSStringFromStdString(const std::string& stdString);
std::string StdStringFromNSString(NSString* nsString);

// Return thread ID as a string.
std::string GetThreadId();

// Return thread ID as string suitable for debug logging.
std::string GetThreadInfo();

// Returns [NSThread currentThread] description as string.
// Example: <NSThread: 0x170066d80>{number = 1, name = main}
std::string GetCurrentThreadDescription();

#if defined(WEBRTC_IOS)
// Returns the current name of the operating system.
std::string GetSystemName();

// Returns the current version of the operating system as a string.
std::string GetSystemVersionAsString();

// Returns the version of the operating system in double representation.
// Uses a cached value of the system version.
double GetSystemVersion();

// Returns the device type.
// Examples: â€iPhoneâ€ and â€iPod touchâ€.
std::string GetDeviceType();
#endif  // defined(WEBRTC_IOS)

// Returns a more detailed device name.
// Examples: "iPhone 5s (GSM)" and "iPhone 6 Plus".
std::string GetDeviceName();

// Returns the name of the process. Does not uniquely identify the process.
std::string GetProcessName();

// Returns the identifier of the process (often called process ID).
int GetProcessID();

// Returns a string containing the version of the operating system on which the
// process is executing. The string is string is human readable, localized, and
// is appropriate for displaying to the user.
std::string GetOSVersionString();

// Returns the number of processing cores available on the device.
int GetProcessorCount();

#if defined(WEBRTC_IOS)
// Indicates whether Low Power Mode is enabled on the iOS device.
bool GetLowPowerModeEnabled();
#endif

}  // namespace ios
}  // namespace webrtc

#endif  // SDK_OBJC_NATIVE_SRC_AUDIO_HELPERS_H_
