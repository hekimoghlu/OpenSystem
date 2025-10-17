/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 18, 2024.
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
/*
 * Copyright (c) 1999 Apple Computer, Inc.  All rights reserved.
 *
 * HISTORY
 *
 */

#ifndef _IOKIT_IONOTIFIER_H
#define _IOKIT_IONOTIFIER_H

#include <libkern/c++/OSObject.h>

/*! @class IONotifier : public OSObject
 *   @abstract An abstract base class defining common methods for controlling a notification request.
 *   @discussion IOService notification requests are represented as implementations of the IONotifier object. It defines methods to enable, disable and remove notification requests. These actions are synchronized with invocations of the notification handler, so removing a notification request will guarantee the handler is not being executed. */

class IONotifier : public OSObject
{
	OSDeclareAbstractStructors(IONotifier);

public:

/*! @function remove
 *   @abstract Removes the notification request and releases it.
 *   @discussion Removes the notification request and release it. Since creating an IONotifier instance will leave it with a retain count of one, creating an IONotifier and then removing it will destroy it. This method is synchronous with any handler invocations, so when this method returns its guaranteed the handler will not be in entered. */

	virtual void remove() = 0;

/*! @function disable
 *   @abstract Disables the notification request.
 *   @discussion Disables the notification request. This method is synchronous with any handler invocations, so when this method returns its guaranteed the handler will not be in entered.
 *   @result Returns the previous enable state of the IONotifier. */

	virtual bool disable() = 0;

/*! @function enable
 *   @abstract Sets the enable state of the notification request.
 *   @discussion Restores the enable state of the notification request, given the previous state passed in.
 *   @param was The enable state of the notifier to restore. */

	virtual void enable( bool was ) = 0;
};

#endif /* ! _IOKIT_IONOTIFIER_H */
