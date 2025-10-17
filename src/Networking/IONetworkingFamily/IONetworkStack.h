/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 2, 2024.
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
#ifndef _IONETWORKSTACK_H
#define _IONETWORKSTACK_H

// User-client keys
//
#define kIONetworkStackUserCommandKey   "IONetworkStackUserCommand"
#define kIONetworkStackUserCommand      "IONetworkStackUserCommand"

enum {
    kIONetworkStackRegisterInterfaceWithUnit        = 0,
    kIONetworkStackRegisterInterfaceWithLowestUnit  = 1,
    kIONetworkStackRegisterInterfaceAll             = 2
};

#ifdef KERNEL

class IONetworkInterface;

class IONetworkStack : public IOService
{
    OSDeclareFinalStructors( IONetworkStack )

protected:
    OSSet *             _ifListNaming;
    OSArray *           _ifListDetach;
    OSArray *           _ifListAttach;
    OSDictionary *      _ifPrefixDict;
    IONotifier *        _ifNotifier;
    IORecursiveLock *   _stateLock;
    thread_call_t       _asyncThread;
    const OSSymbol *    _noBSDAttachSymbol;
    IONotifier *        _sleepWakeNotifier;

    static SInt32       orderNetworkInterfaces(
                            const OSMetaClassBase * obj1,
                            const OSMetaClassBase * obj2,
                            void *                  ref );

    virtual void        free( void ) APPLE_KEXT_OVERRIDE;

    bool                interfacePublished(
                            void *          refCon,
                            IOService *     service,
                            IONotifier *    notifier );

    void                asyncWork( void );

    bool                insertNetworkInterface(
                            IONetworkInterface * netif );

    void                removeNetworkInterface(
                            IONetworkInterface * netif );

    uint32_t            getNextAvailableUnitNumber(
                            const char *         name,
                            uint32_t             startingUnit );

    bool                reserveInterfaceUnitNumber(
                            IONetworkInterface * netif,
                            uint32_t             unit,
                            bool                 isUnitFixed,
                            bool *               attachToBSD );

    IOReturn            attachNetworkInterfaceToBSD(
                            IONetworkInterface * netif );

    IOReturn            registerAllNetworkInterfaces( void );

    IOReturn            registerNetworkInterface(
                            IONetworkInterface * netif,
                            uint32_t             unit,
                            bool                 isUnitFixed );

    static IOReturn     handleSystemSleep(
                            void * target, void * refCon,
                            UInt32 messageType, IOService * provider,
                            void * messageArgument, vm_size_t argSize );

public:
    virtual bool        start( IOService * provider ) APPLE_KEXT_OVERRIDE;
    virtual void        stop( IOService * provider ) APPLE_KEXT_OVERRIDE;
    virtual bool        finalize( IOOptionBits options ) APPLE_KEXT_OVERRIDE;
    virtual bool        didTerminate( IOService *, IOOptionBits, bool * ) APPLE_KEXT_OVERRIDE;
    virtual IOReturn    setProperties( OSObject * properties ) APPLE_KEXT_OVERRIDE;
};

#endif /* KERNEL */
#endif /* !_IONETWORKSTACK_H */
