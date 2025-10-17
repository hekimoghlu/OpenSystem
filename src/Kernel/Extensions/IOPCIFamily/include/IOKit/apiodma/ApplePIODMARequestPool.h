/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 20, 2022.
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

//
//  ApplePIODMARequestPool.h
//  ApplePIODMA
//
//  Created by Kevin Strasberg on 6/26/20.
//

#ifndef ApplePIODMARequestPool_h
#define ApplePIODMARequestPool_h
#include <IOKit/IOCommand.h>
#include <IOKit/IODMACommand.h>
#include <IOKit/IOCommandPool.h>
#include <IOKit/IOWorkLoop.h>
#include <IOKit/apiodma/ApplePIODMADefinitions.h>

class ApplePIODMARequestPool :  public IOCommandPool
{
    OSDeclareDefaultStructors(ApplePIODMARequestPool);

public:
public:
	static ApplePIODMARequestPool*	withWorkLoop(IOWorkLoop* workLoop,
                                                 IOMapper*   mapper,
                                                 uint32_t    maxOutstandingCommands,
                                                 uint32_t    byteAlignment,
                                                 uint8_t     numberOfAddressBits,
                                                 uint64_t    maxTransferSize,
                                                 uint64_t    maxSegmentSize);

    virtual void free() override;
    unsigned int maximumCommandsSupported();

protected:
    virtual bool initWithWorkLoop(IOWorkLoop* workLoop,
                                  IOMapper*   mapper,
                                  uint32_t    maxOutstandingCommands,
                                  uint32_t    byteAlignment,
                                  uint8_t     numberOfAddressBits,
                                  uint64_t    maxTransferSize,
                                  uint64_t    maxSegmentSize);

#pragma mark Pool Management

    virtual IOCommand* allocateCommand();

    uint32_t    _debugLoggingMask;
    IOWorkLoop* _workLoop;
    IOMapper*   _memoryMapper;
    uint32_t    _maxOutstandingCommands;
    uint32_t    _byteAlignment;
    uint8_t     _numberOfAddressBits;
    uint64_t    _maxTransferSize;
    uint64_t    _maxSegmentSize;
};

#endif /* ApplePIODMARequestPool_h */
