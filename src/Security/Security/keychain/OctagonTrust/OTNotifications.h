/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 15, 2021.
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
#if __OBJC2__

#ifndef OTNOTIFICATIONS_H
#define OTNOTIFICATIONS_H

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*
 * This is a Darwin notification sent when the user controllable view syncing status changes.
 */
extern NSString* OTUserControllableViewStatusChanged;

/*
 * This is a Darwin notification send whenever any peer (remote or local) updates any Octagon
 * data. Note that this will be sent even if the local device isn't trusted in Octagon, or possibly
 * even if the local device is locked and cannot respond to the change.
 */
extern NSString* OTCliqueChanged;

/*
 * This is a Darwin notification sent when an Octagon recovery happens using the private key
 * material of another device. Importantly, after this is sent, there is no expectation that
 * there are other devices online that might have any keychain or other user secret data
 * to send to this device.
 */
extern NSString* OTJoinedViaBottle;

NS_ASSUME_NONNULL_END

#endif // OTNOTIFICATIONS_H
#endif
