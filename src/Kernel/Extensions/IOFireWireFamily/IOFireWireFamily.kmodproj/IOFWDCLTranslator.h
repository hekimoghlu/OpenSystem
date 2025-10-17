/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 31, 2024.
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
 * A DCL program to interpret (in software) a program that's too complicated
 * for the DMA engine.
 *
 * HISTORY
 *
 */


#ifndef _IOKIT_IOFWDCLTRANSLATOR_H
#define _IOKIT_IOFWDCLTRANSLATOR_H

#include <libkern/c++/OSObject.h>
#include <IOKit/firewire/IOFWDCLProgram.h>


/*! @class IODCLTranslator
*/

class IODCLTranslator : public IODCLProgram
{
    OSDeclareAbstractStructors(IODCLTranslator)

protected:
    enum
    {
        kNumPingPongs				= 2,
        kNumPacketsPerPingPong		= 500,
        kMaxIsochPacketSize			= 1000,
        kPingPongBufferSize			= kNumPingPongs * kNumPacketsPerPingPong * kMaxIsochPacketSize
    };

    // Opcodes and buffer for pingpong program
    DCLLabel			fStartLabel;
    DCLTransferPacket	fTransfers[kNumPingPongs*kNumPacketsPerPingPong];
    DCLCallProc			fCalls[kNumPingPongs];
    DCLJump				fJumpToStart;
    UInt8				fBuffer[kPingPongBufferSize];

    IODCLProgram *		fHWProgram;				// Hardware program executing our opcodes
    DCLCommand*			fToInterpret;			// The commands to interpret
    DCLCommand*			fCurrentDCLCommand;		// Current command to interpret
    int					fPingCount;				// Are we pinging or ponging?
    UInt32				fPacketHeader;

    static void ListeningDCLPingPongProc(DCLCommand* pDCLCommand);
    static void TalkingDCLPingPongProc(DCLCommand* pDCLCommand);

public:
    virtual bool init(DCLCommand* toInterpret);
    virtual IOReturn allocateHW(IOFWSpeed speed, UInt32 chan) APPLE_KEXT_OVERRIDE;
    virtual IOReturn releaseHW() APPLE_KEXT_OVERRIDE;
    virtual IOReturn notify(IOFWDCLNotificationType notificationType,
	DCLCommand** dclCommandList, UInt32 numDCLCommands) APPLE_KEXT_OVERRIDE;
    virtual void stop() APPLE_KEXT_OVERRIDE;

    DCLCommand* getTranslatorOpcodes();
    void setHWProgram(IODCLProgram *program);
};

/*! @class IODCLTranslateTalk
*/

class IODCLTranslateTalk : public IODCLTranslator
{
    OSDeclareDefaultStructors(IODCLTranslateTalk)

protected:

public:
    virtual IOReturn compile(IOFWSpeed speed, UInt32 chan) APPLE_KEXT_OVERRIDE;
    virtual IOReturn start() APPLE_KEXT_OVERRIDE;

};

/*! @class IODCLTranslateListen
*/

class IODCLTranslateListen : public IODCLTranslator
{
    OSDeclareDefaultStructors(IODCLTranslateListen)

protected:

public:
    virtual IOReturn compile(IOFWSpeed speed, UInt32 chan) APPLE_KEXT_OVERRIDE;
    virtual IOReturn start() APPLE_KEXT_OVERRIDE;

};
#endif /* ! _IOKIT_IOFWDCLPROGRAM_H */

