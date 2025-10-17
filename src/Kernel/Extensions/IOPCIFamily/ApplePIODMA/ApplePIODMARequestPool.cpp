/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 19, 2023.
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
//  ApplePIODMARequestPool.cpp
//  ApplePIODMA
//
//  Created by Kevin Strasberg on 6/26/20.
//
#include <IOKit/IOReturn.h>
#include <IOKit/apiodma/ApplePIODMADebug.h>
#include <IOKit/apiodma/ApplePIODMARequest.h>
#include <IOKit/apiodma/ApplePIODMARequestPool.h>


#define super IOCommandPool
OSDefineMetaClassAndStructors(ApplePIODMARequestPool, IOCommandPool)

#define debug(mask, fmt, args...) pioDMADebugObjectWithClass(mask, ApplePIODMARequestPool, fmt,##args)

ApplePIODMARequestPool * ApplePIODMARequestPool::withWorkLoop(IOWorkLoop * workLoop,
                                                              IOMapper *   mapper,
                                                              uint32_t maxOutstandingCommands,
                                                              uint32_t byteAlignment,
                                                              uint8_t numberOfAddressBits,
                                                              uint64_t maxTransferSize,
                                                              uint64_t maxSegmentSize)
{
    ApplePIODMARequestPool* result = new ApplePIODMARequestPool;
    if(   result != NULL
       && result->initWithWorkLoop(workLoop,
                                   mapper,
                                   maxOutstandingCommands,
                                   byteAlignment,
                                   numberOfAddressBits,
                                   maxTransferSize,
                                   maxSegmentSize) == false)
    {
        OSSafeReleaseNULL(result);
    }

    return result;
}

#pragma mark IOCommandPool overrides
bool ApplePIODMARequestPool::initWithWorkLoop(IOWorkLoop* workLoop,
                                              IOMapper*   mapper,
                                              uint32_t    maxOutstandingCommands,
                                              uint32_t    byteAlignment,
                                              uint8_t     numberOfAddressBits,
                                              uint64_t    maxTransferSize,
                                              uint64_t    maxSegmentSize)
{
    if(super::initWithWorkLoop(workLoop) == false)
    {
        return false;
    }

    _debugLoggingMask = applePIODMAgetDebugLoggingMaskForMetaClass(getMetaClass(), super::metaClass);

    _maxOutstandingCommands = maxOutstandingCommands;
    _byteAlignment          = byteAlignment;
    _numberOfAddressBits    = numberOfAddressBits;
    _maxTransferSize        = maxTransferSize;
    _maxSegmentSize         = maxSegmentSize;
    _maxOutstandingCommands = maxOutstandingCommands;

    _workLoop = workLoop;
    _workLoop->retain();

    if(mapper != NULL)
    {
        _memoryMapper = mapper;
        _memoryMapper->retain();
    }

    // allocate the commands during initialization
    for(unsigned int i = 0; i < _maxOutstandingCommands; i++)
    {
        IOCommand* command = allocateCommand();
        returnCommand(command);
    }


    return true;
}

void ApplePIODMARequestPool::free()
{
    OSSafeReleaseNULL(_workLoop);
    OSSafeReleaseNULL(_memoryMapper);
    super::free();
}

#pragma mark Pool Management
IOCommand* ApplePIODMARequestPool::allocateCommand()
{
    return ApplePIODMARequest::withMapper(_memoryMapper,
                                          _byteAlignment,
                                          _numberOfAddressBits,
                                          _maxTransferSize,
                                          _maxSegmentSize);
}
