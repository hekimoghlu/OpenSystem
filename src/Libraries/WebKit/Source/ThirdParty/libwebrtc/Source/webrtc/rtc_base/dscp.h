/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 23, 2022.
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
#ifndef RTC_BASE_DSCP_H_
#define RTC_BASE_DSCP_H_

namespace rtc {
// Differentiated Services Code Point.
// See http://tools.ietf.org/html/rfc2474 for details.
enum DiffServCodePoint {
  DSCP_NO_CHANGE = -1,
  DSCP_DEFAULT = 0,  // Same as DSCP_CS0
  DSCP_CS0 = 0,      // The default
  DSCP_CS1 = 8,      // Bulk/background traffic
  DSCP_AF11 = 10,
  DSCP_AF12 = 12,
  DSCP_AF13 = 14,
  DSCP_CS2 = 16,
  DSCP_AF21 = 18,
  DSCP_AF22 = 20,
  DSCP_AF23 = 22,
  DSCP_CS3 = 24,
  DSCP_AF31 = 26,
  DSCP_AF32 = 28,
  DSCP_AF33 = 30,
  DSCP_CS4 = 32,
  DSCP_AF41 = 34,  // Video
  DSCP_AF42 = 36,  // Video
  DSCP_AF43 = 38,  // Video
  DSCP_CS5 = 40,   // Video
  DSCP_EF = 46,    // Voice
  DSCP_CS6 = 48,   // Voice
  DSCP_CS7 = 56,   // Control messages
};

}  // namespace rtc

#endif  // RTC_BASE_DSCP_H_
