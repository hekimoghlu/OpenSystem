/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 14, 2023.
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
#ifndef _IOKIT_APPLE8259PIC_H
#define _IOKIT_APPLE8259PIC_H

#include <IOKit/IOInterrupts.h>
#include <IOKit/IOInterruptController.h>
#include <libkern/c++/OSContainers.h>
#include <architecture/i386/pio.h>

#define kClockIRQ              0

#define kPIC1BasePort          0x20
#define kPIC2BasePort          0xa0

#define kPIC1TriggerTypePort   0x4d0
#define kPIC2TriggerTypePort   0x4d1

#define kPICCmdPortOffset      0
#define kPICDataPortOffset     1

#define kEOICommand            0x20

#define kPICSlaveID            2       // Slave ID for second PIC

#define kNumVectors            16

#define IS_SLAVE_VECTOR(x)     ((x) & 8)

// ICW1
//
#define kPIC_ICW1(x)           ((x) + kPICCmdPortOffset)
#define kPIC_ICW1_MBO          0x10    // must be one
#define kPIC_ICW1_LTIM         0x08    // level/edge triggered mode
#define kPIC_ICW1_ADI          0x04    // 4/8 byte call address interval
#define kPIC_ICW1_SNGL         0x02    // single/cascade mode
#define kPIC_ICW1_IC4          0x01    // ICW4 needed/not needed

// ICW2 - Interrupt vector address (bits 7 - 3).
//
#define kPIC_ICW2(x)           ((x) + kPICDataPortOffset)

// ICW3 - Slave device.
//
#define kPIC_ICW3(x)           ((x) + kPICDataPortOffset)

// ICW4
//
#define kPIC_ICW4(x)           ((x) + kPICDataPortOffset)
#define kPIC_ICW4_SFNM         0x10    // special fully nested mode
#define kPIC_ICW4_BUF          0x08    // buffered mode
#define kPIC_ICW4_MS           0x04    // master/slave
#define kPIC_ICW4_AEOI         0x02    // automatic end of interrupt mode
#define kPIC_ICW4_uPM          0x01    // 8088 (vs. 8085) operation

// OCW1 - Interrupt mask.
//
#define kPIC_OCW1(x)           ((x) + kPICDataPortOffset)

// OCW2 - Bit 4 must be zero.
//
#define kPIC_OCW2(x)           ((x) + kPICCmdPortOffset)
#define kPIC_OCW2_R            0x80    // rotation
#define kPIC_OCW2_SL           0x40    // specific
#define kPIC_OCW2_EOI          0x20
#define kPIC_OCW2_LEVEL(x)     ((x) & 0x07)

// OCW3 - Bit 4 must be zero.
//
#define kPIC_OCW3(x)           ((x) + kPICCmdPortOffset)
#define kPIC_OCW3_ESMM         0x40    // special mask mode
#define kPIC_OCW3_SMM          0x20
#define kPIC_OCW3_MBO          0x08    // must be one
#define kPIC_OCW3_P            0x04    // poll
#define kPIC_OCW3_RR           0x02
#define kPIC_OCW3_RIS          0x01


#define Apple8259PIC Apple8259PICInterruptController

class Apple8259PIC : public IOInterruptController
{
    OSDeclareDefaultStructors( Apple8259PICInterruptController );

protected:
    UInt16           _interruptMasks;
    UInt16           _interruptTriggerTypes;
    IOSimpleLock *   _interruptLock;
    const OSSymbol * _handleSleepWakeFunction;

    inline void writeInterruptMask( long irq )
    {
        kprintf("Apple8259PIC::writeInterruptMask(%lu)\n", irq);
        if ( IS_SLAVE_VECTOR(irq) )
            outb( kPIC_OCW1(kPIC2BasePort), _interruptMasks >> 8 );
        else
            outb( kPIC_OCW1(kPIC1BasePort), _interruptMasks & 0xff );
    }

    inline void disableInterrupt( long irq )
    {
        kprintf("Apple8259PIC::disableInterrupt(%lu)", irq);
        _interruptMasks |= (1 << irq);
        writeInterruptMask(irq);
    }

    inline void enableInterrupt( long irq )
    {
        kprintf("Apple8259PIC::enableInterrupt(%lu)", irq);
        _interruptMasks &= ~(1 << irq);
        writeInterruptMask(irq);
    }

    inline void ackInterrupt( long irq )
    {
        if ( IS_SLAVE_VECTOR(irq) )
            outb( kPIC_OCW2(kPIC2BasePort), kEOICommand );
        outb( kPIC_OCW2(kPIC1BasePort), kEOICommand );
    }
    
    inline int   getTriggerType(long irq)
    {
        return ( _interruptTriggerTypes & (1 << irq) ) ?
        kIOInterruptTypeLevel : kIOInterruptTypeEdge;
    }

    virtual void     initializePIC( UInt16 port,
                                    UInt8 icw1, UInt8 icw2,
                                    UInt8 icw3, UInt8 icw4 );

    virtual void     prepareForSleep( void );

    virtual void     resumeFromSleep( void );

    virtual void     free( void );

public:
    virtual bool     start( IOService * provider );

//    virtual IOReturn getInterruptType( IOService * nub,
//                                       int   source,
//                                       int * interruptType );

    virtual int      getVectorType( long vectorNumber,
                                    IOInterruptVector * vector);

    virtual IOInterruptAction getInterruptHandlerAddress( void );

    virtual IOReturn handleInterrupt( void * refCon,
                                      IOService * nub,
                                      int source );

    virtual bool     vectorCanBeShared( long vectorNumber,
                                        IOInterruptVector * vector );

    virtual void     initVector( long vectorNumber,
                                 IOInterruptVector * vector );

    virtual void     disableVectorHard( long vectorNumber,
                                        IOInterruptVector * vector );

    virtual void     enableVector( long vectorNumber,
                                   IOInterruptVector * vector );

    virtual IOReturn callPlatformFunction( const OSSymbol * function,
                                           bool waitForFunction,
                                           void * param1, void * param2,
                                           void * param3, void * param4 );
};

#endif /* !_IOKIT_APPLE8259PIC_H */
