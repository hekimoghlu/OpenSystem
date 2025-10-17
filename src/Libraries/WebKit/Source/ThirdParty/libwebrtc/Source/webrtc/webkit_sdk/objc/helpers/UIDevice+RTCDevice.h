/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 1, 2022.
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
#if !defined(WEBRTC_WEBKIT_BUILD)

#import <UIKit/UIKit.h>

typedef NS_ENUM(NSInteger, RTCDeviceType) {
  RTCDeviceTypeUnknown,
  RTCDeviceTypeIPhone1G,
  RTCDeviceTypeIPhone3G,
  RTCDeviceTypeIPhone3GS,
  RTCDeviceTypeIPhone4,
  RTCDeviceTypeIPhone4Verizon,
  RTCDeviceTypeIPhone4S,
  RTCDeviceTypeIPhone5GSM,
  RTCDeviceTypeIPhone5GSM_CDMA,
  RTCDeviceTypeIPhone5CGSM,
  RTCDeviceTypeIPhone5CGSM_CDMA,
  RTCDeviceTypeIPhone5SGSM,
  RTCDeviceTypeIPhone5SGSM_CDMA,
  RTCDeviceTypeIPhone6Plus,
  RTCDeviceTypeIPhone6,
  RTCDeviceTypeIPhone6S,
  RTCDeviceTypeIPhone6SPlus,
  RTCDeviceTypeIPhone7,
  RTCDeviceTypeIPhone7Plus,
  RTCDeviceTypeIPhoneSE,
  RTCDeviceTypeIPhone8,
  RTCDeviceTypeIPhone8Plus,
  RTCDeviceTypeIPhoneX,
  RTCDeviceTypeIPhoneXS,
  RTCDeviceTypeIPhoneXSMax,
  RTCDeviceTypeIPhoneXR,
  RTCDeviceTypeIPhone11,
  RTCDeviceTypeIPhone11Pro,
  RTCDeviceTypeIPhone11ProMax,
  RTCDeviceTypeIPodTouch1G,
  RTCDeviceTypeIPodTouch2G,
  RTCDeviceTypeIPodTouch3G,
  RTCDeviceTypeIPodTouch4G,
  RTCDeviceTypeIPodTouch5G,
  RTCDeviceTypeIPodTouch6G,
  RTCDeviceTypeIPodTouch7G,
  RTCDeviceTypeIPad,
  RTCDeviceTypeIPad2Wifi,
  RTCDeviceTypeIPad2GSM,
  RTCDeviceTypeIPad2CDMA,
  RTCDeviceTypeIPad2Wifi2,
  RTCDeviceTypeIPadMiniWifi,
  RTCDeviceTypeIPadMiniGSM,
  RTCDeviceTypeIPadMiniGSM_CDMA,
  RTCDeviceTypeIPad3Wifi,
  RTCDeviceTypeIPad3GSM_CDMA,
  RTCDeviceTypeIPad3GSM,
  RTCDeviceTypeIPad4Wifi,
  RTCDeviceTypeIPad4GSM,
  RTCDeviceTypeIPad4GSM_CDMA,
  RTCDeviceTypeIPad5,
  RTCDeviceTypeIPad6,
  RTCDeviceTypeIPadAirWifi,
  RTCDeviceTypeIPadAirCellular,
  RTCDeviceTypeIPadAirWifiCellular,
  RTCDeviceTypeIPadAir2,
  RTCDeviceTypeIPadMini2GWifi,
  RTCDeviceTypeIPadMini2GCellular,
  RTCDeviceTypeIPadMini2GWifiCellular,
  RTCDeviceTypeIPadMini3,
  RTCDeviceTypeIPadMini4,
  RTCDeviceTypeIPadPro9Inch,
  RTCDeviceTypeIPadPro12Inch,
  RTCDeviceTypeIPadPro12Inch2,
  RTCDeviceTypeIPadPro10Inch,
  RTCDeviceTypeIPad7Gen10Inch,
  RTCDeviceTypeIPadPro3Gen11Inch,
  RTCDeviceTypeIPadPro3Gen12Inch,
  RTCDeviceTypeIPadMini5Gen,
  RTCDeviceTypeIPadAir3Gen,
  RTCDeviceTypeSimulatori386,
  RTCDeviceTypeSimulatorx86_64,
};

@interface UIDevice (RTCDevice)

+ (RTCDeviceType)deviceType;
+ (BOOL)isIOS11OrLater;

@end

#endif
