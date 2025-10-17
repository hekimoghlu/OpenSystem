/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 29, 2023.
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
 * Copyright (c) 1999-2002 Apple Computer, Inc.  All rights reserved.
 *
 * IOFWIsochPort is an abstract object that represents hardware on the bus
 * (locally or remotely) that sends or receives isochronous packets.
 * Local ports are implemented by the local device driver,
 * Remote ports are implemented by the driver for the remote device.
 *
 * HISTORY
 *
 */

#import "IOFWIsochPort.h"


OSDefineMetaClass( IOFWIsochPort, OSObject )
OSDefineAbstractStructors(IOFWIsochPort, OSObject)
OSMetaClassDefineReservedUnused(IOFWIsochPort, 0);
OSMetaClassDefineReservedUnused(IOFWIsochPort, 1);
