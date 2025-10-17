/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 27, 2023.
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

#if __has_attribute(enum_extensibility)
#define __CF_ENUM_ATTRIBUTES __attribute__((enum_extensibility(open)))
#define __CF_CLOSED_ENUM_ATTRIBUTES __attribute__((enum_extensibility(closed)))
#define __CF_OPTIONS_ATTRIBUTES __attribute__((flag_enum,enum_extensibility(open)))
#else
#define __CF_ENUM_ATTRIBUTES
#define __CF_CLOSED_ENUM_ATTRIBUTES
#define __CF_OPTIONS_ATTRIBUTES
#endif

#define CF_OPTIONS(_type, _name) _type __attribute__((availability(language, unavailable))) _name; enum __CF_OPTIONS_ATTRIBUTES : _name
#define NS_OPTIONS(_type, _name) CF_OPTIONS(_type, _name)
#define UIKIT_EXTERN extern "C" __attribute__((visibility("default")))

typedef long NSInteger;

UIKIT_EXTERN
@interface UIPrinter

typedef NS_OPTIONS(NSInteger, UIPrinterJobTypes) {
  UIPrinterJobTypeUnknown = 0,
  UIPrinterJobTypeDocument = 1 << 0,
  UIPrinterJobTypeEnvelope = 1 << 1,
  UIPrinterJobTypeLabel = 1 << 2,
  UIPrinterJobTypePhoto = 1 << 3,
  UIPrinterJobTypeReceipt = 1 << 4,
  UIPrinterJobTypeRoll = 1 << 5,
  UIPrinterJobTypeLargeFormat = 1 << 6,
  UIPrinterJobTypePostcard = 1 << 7
};

@end
