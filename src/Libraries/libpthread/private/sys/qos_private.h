/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 26, 2022.
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
#ifndef _QOS_SYS_PRIVATE_H
#define _QOS_SYS_PRIVATE_H

/*! 
 * @constant QOS_CLASS_MAINTENANCE
 * @abstract A QOS class which indicates work performed by this thread was not
 * initiated by the user and that the user may be unaware of the results.
 * @discussion Such work is requested to run at a priority far below other work
 * including significant I/O throttling. The use of this QOS class indicates
 * the work should be run in the most energy and thermally-efficient manner
 * possible, and may be deferred for a long time in order to preserve
 * system responsiveness for the user.
 * This is SPI for use by Spotlight and Time Machine only.
 */
#define QOS_CLASS_MAINTENANCE	((qos_class_t)0x05)

#endif //_QOS_SYS_PRIVATE_H
