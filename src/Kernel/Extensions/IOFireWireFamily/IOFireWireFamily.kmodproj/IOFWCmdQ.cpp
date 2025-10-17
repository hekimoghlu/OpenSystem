/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 19, 2025.
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
#import <IOKit/firewire/IOFireWireController.h>

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// executeQueue
//
//

bool IOFWCmdQ::executeQueue(bool all)
{
    IOFWCommand *cmd;
    
	cmd = fHead;
    
	while( cmd ) 
	{
        IOFWCommand *newHead;
        newHead = cmd->getNext();
        
		if( newHead )
            newHead->fQueuePrev = NULL;
        else
            fTail = NULL;
        
		fHead = newHead;

        cmd->fQueue = NULL;	// Not on this queue anymore
		cmd->startExecution();
        
		if(!all)
            break;
        
		cmd = newHead;
    }
	
    return fHead != NULL;	// ie. more to do
}

// checkProgress
//
//

void IOFWCmdQ::checkProgress( void )
{
    IOFWCommand *cmd;
    cmd = fHead;
    while(cmd) 
	{
        IOFWCommand *next;
        next = cmd->getNext();

		// see if this command has gone on for too long
		IOReturn status = cmd->checkProgress();
		if( status != kIOReturnSuccess )
		{
            cmd->complete( status );
        }
        cmd = next;
    }
}

// headChanged
//
//

void IOFWCmdQ::headChanged(IOFWCommand *oldHead)
{
    
}

