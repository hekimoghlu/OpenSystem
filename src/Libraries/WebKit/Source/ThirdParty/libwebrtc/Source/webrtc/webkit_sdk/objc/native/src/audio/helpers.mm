/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 20, 2025.
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
#import <Foundation/Foundation.h>
#import <sys/sysctl.h>
#if defined(WEBRTC_IOS)
#import <UIKit/UIKit.h>
#endif

#include <memory>

#include "helpers.h"
#include "rtc_base/checks.h"
#include "rtc_base/logging.h"

namespace webrtc {
namespace ios {

NSString* NSStringFromStdString(const std::string& stdString) {
  // std::string may contain null termination character so we construct
  // using length.
  return [[NSString alloc] initWithBytes:stdString.data()
                                  length:stdString.length()
                                encoding:NSUTF8StringEncoding];
}

std::string StdStringFromNSString(NSString* nsString) {
  NSData* charData = [nsString dataUsingEncoding:NSUTF8StringEncoding];
  return std::string(reinterpret_cast<const char*>([charData bytes]),
                     [charData length]);
}

bool CheckAndLogError(BOOL success, NSError* error) {
  if (!success) {
    NSString* msg =
        [NSString stringWithFormat:@"Error: %ld, %@, %@", (long)error.code,
                                   error.localizedDescription,
                                   error.localizedFailureReason];
    RTC_LOG(LS_ERROR) << StdStringFromNSString(msg);
    return false;
  }
  return true;
}

// TODO(henrika): see if it is possible to move to GetThreadName in
// platform_thread.h and base it on pthread methods instead.
std::string GetCurrentThreadDescription() {
  NSString* name = [NSString stringWithFormat:@"%@", [NSThread currentThread]];
  return StdStringFromNSString(name);
}

#if defined(WEBRTC_IOS)
std::string GetSystemName() {
  NSString* osName = [[UIDevice currentDevice] systemName];
  return StdStringFromNSString(osName);
}

std::string GetSystemVersionAsString() {
  NSString* osVersion = [[UIDevice currentDevice] systemVersion];
  return StdStringFromNSString(osVersion);
}

std::string GetDeviceType() {
  NSString* deviceModel = [[UIDevice currentDevice] model];
  return StdStringFromNSString(deviceModel);
}

bool GetLowPowerModeEnabled() {
  return [NSProcessInfo processInfo].lowPowerModeEnabled;
}
#endif

std::string GetDeviceName() {
  size_t size;
  sysctlbyname("hw.machine", NULL, &size, NULL, 0);
  std::unique_ptr<char[]> machine;
  machine.reset(new char[size]);
  sysctlbyname("hw.machine", machine.get(), &size, NULL, 0);
  return std::string(machine.get());
}

std::string GetProcessName() {
  NSString* processName = [NSProcessInfo processInfo].processName;
  return StdStringFromNSString(processName);
}

int GetProcessID() {
  return [NSProcessInfo processInfo].processIdentifier;
}

std::string GetOSVersionString() {
  NSString* osVersion =
      [NSProcessInfo processInfo].operatingSystemVersionString;
  return StdStringFromNSString(osVersion);
}

int GetProcessorCount() {
  return [NSProcessInfo processInfo].processorCount;
}

}  // namespace ios
}  // namespace webrtc
