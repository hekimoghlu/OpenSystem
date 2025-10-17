/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 6, 2023.
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
#ifndef _IOGATEDOUTPUTQUEUE_H
#define _IOGATEDOUTPUTQUEUE_H

#include <IOKit/IOWorkLoop.h>
#include <IOKit/IOCommandGate.h>
#include <IOKit/IOInterruptEventSource.h>
#include <IOKit/network/IOBasicOutputQueue.h>

/*! @class IOGatedOutputQueue
    @abstract An extension of an IOBasicOutputQueue. 
    @discussion An IOCommandGate
    object is created by this queue and added to a work loop as an
    event source. All calls to the target by the consumer thread must
    occur with the gate closed. Therefore, all calls to the target of
    this type of queue will be serialized with any other thread that
    runs on the same work loop context. This is useful for network
    drivers that have a tight hardware coupling between the transmit
    and receive engines, and a single-threaded hardware access model
    is desirable. 
*/


class __exported IOGatedOutputQueue : public IOBasicOutputQueue
{
    OSDeclareDefaultStructors( IOGatedOutputQueue )

private:
    static void gatedOutput(OSObject *           owner,
                            IOGatedOutputQueue * self,
                            IOMbufQueue *        queue,
                            UInt32 *             state);

    static void restartDeferredOutput(OSObject *               owner,
                                      IOInterruptEventSource * sender,
                                      int                      count);

protected:
    IOCommandGate *           _gate;
    IOInterruptEventSource *  _interruptSrc;

/*! @function output
    @abstract Transfers all packets in the mbuf queue to the target.
    @param queue A queue of output packets.
    @param state Return a state bit defined by IOBasicOutputQueue that
    declares the new state of the queue following this method call.
    A kStateStalled is returned if the queue should stall, otherwise 0
    is returned. 
*/

    virtual void output(IOMbufQueue * queue, UInt32 * state) APPLE_KEXT_OVERRIDE;

/*! @function free
    @abstract Frees the IOGatedOutputQueue object.
    @discussion Release allocated resources, then call super::free(). */

    virtual void free() APPLE_KEXT_OVERRIDE;

/*! @function output
    @abstract Overrides the method inherited from IOOutputQueue.
    @result Returns true if a thread was successfully scheduled to service
    the queue. 
*/

    virtual bool scheduleServiceThread(void * param) APPLE_KEXT_OVERRIDE;

public:

/*! @function init
    @abstract Initializes an IOGatedOutputQueue object.
    @param target The object that will handle packets removed from the
    queue, and is usually a subclass of IONetworkController.
    @param action The function that will handle packets removed from the
    queue.
    @param workloop A workloop object. An IOCommandGate object is created
    and added to this workloop as an event source.
    @param capacity The initial capacity of the output queue.
    @result Returns true if initialized successfully, false otherwise. 
*/

    virtual bool init(OSObject *     target,
                      IOOutputAction action,
                      IOWorkLoop *   workloop,
                      UInt32         capacity = 0,
                      UInt32         priorities = 1);

/*! @function withTarget
    @abstract Factory method that constructs and initializes an
    IOGatedOutputQueue object.
    @param target An IONetworkController object that will handle packets
    removed from the queue.
    @param workloop A workloop object. An IOCommandGate object is created
    and added to this workloop as an event source.
    @param capacity The initial capacity of the output queue.
    @result Returns an IOGatedOutputQueue object on success, or 0 otherwise. 
*/

    static IOGatedOutputQueue * withTarget(IONetworkController * target,
                                           IOWorkLoop *          workloop,
                                           UInt32                capacity = 0);

    /*! @function withTarget
     @abstract Factory method that constructs and initializes an
     IOGatedOutputQueue object.
     @param target An IONetworkController object that will handle packets
     removed from the queue.
     @param workloop A workloop object. An IOCommandGate object is created
     and added to this workloop as an event source.
     @param capacity The initial capacity of the output queue.
     @param priorities The number of traffic priorities supported
     @result Returns an IOGatedOutputQueue object on success, or 0 otherwise. 
     */
    
    static IOGatedOutputQueue * withTarget(IONetworkController * target,
                                           IOWorkLoop *          workloop,
                                           UInt32                capacity,
                                           UInt32                priorities);
    
/*! @function withTarget
    @abstract Factory method that constructs and initializes an
    IOGatedOutputQueue object.
    @param target The object that will handle packets removed from the
    queue.
    @param action The function that will handle packets removed from the
    queue.
    @param workloop A workloop object. An IOCommandGate object is created
    and added to this workloop as an event source.
    @param capacity The initial capacity of the output queue.
    @result Returns an IOGatedOutputQueue object on success, or 0 otherwise. 
*/

    static IOGatedOutputQueue * withTarget(OSObject *     target,
                                           IOOutputAction action,
                                           IOWorkLoop *   workloop,
                                           UInt32         capacity = 0);
    
    /*! @function withTarget
     @abstract Factory method that constructs and initializes an
     IOGatedOutputQueue object.
     @param target The object that will handle packets removed from the
     queue.
     @param action The function that will handle packets removed from the
     queue.
     @param workloop A workloop object. An IOCommandGate object is created
     and added to this workloop as an event source.
     @param capacity The initial capacity of the output queue.
     @param priorities The number of traffic priorities supported
     @result Returns an IOGatedOutputQueue object on success, or 0 otherwise. 
     */
    
    static IOGatedOutputQueue * withTarget(OSObject *     target,
                                           IOOutputAction action,
                                           IOWorkLoop *   workloop,
                                           UInt32         capacity,
                                           UInt32         priorities);
};

#endif /* !_IOGATEDOUTPUTQUEUE_H */
