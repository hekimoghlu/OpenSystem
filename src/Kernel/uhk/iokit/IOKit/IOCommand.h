/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 14, 2024.
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
 *
 *	Copyright (c) 2000 Apple Computer, Inc.  All rights reserved.
 *
 *	HISTORY
 *
 *	2001-01-18 gvdl	Made the primary queue pointer public, to be used when
 *			Ownership is clear.
 *	11/13/2000 CJS	Created IOCommand class and implementation
 *
 */

/*!
 * @header IOCommand
 * @abstract
 * This header contains the IOCommand class definition.
 */

#ifndef _IOKIT_IO_COMMAND_H_
#define _IOKIT_IO_COMMAND_H_

/*
 * Kernel
 */

#if defined(KERNEL) && defined(__cplusplus)

#include <kern/queue.h>
#include <libkern/c++/OSObject.h>

/*!
 * @class IOCommand
 * @abstract
 * This class is an abstract class which represents an I/O command.
 * @discussion
 * This class is an abstract class which represents an I/O command passed
 * from a device driver to a controller. All controller commands (e.g. IOATACommand)
 * should inherit from this class.
 */

class IOCommand : public OSObject
{
	OSDeclareDefaultStructors(IOCommand);

public:
	virtual bool init(void) APPLE_KEXT_OVERRIDE;

/*! @var fCommandChain
 *   This variable is used by the current 'owner' to queue the command.  During the life cycle of a command it moves through a series of queues.  This is the queue pointer for it.  Only valid while 'ownership' is clear.  For instance a IOCommandPool uses this pointer to maintain its list of free commands.  May be manipulated using the kern/queue.h macros */
	queue_chain_t fCommandChain;    /* used to queue commands */
};

#endif /* defined(KERNEL) && defined(__cplusplus) */

#endif  /* _IOKIT_IO_COMMAND_H_ */
