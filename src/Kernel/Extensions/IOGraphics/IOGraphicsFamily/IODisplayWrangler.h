/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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
#ifndef _IOKIT_IODISPLAYWRANGLER_H
#define _IOKIT_IODISPLAYWRANGLER_H

#include <IOKit/IOService.h>

#define IOFRAMEBUFFER_PRIVATE
#include <IOKit/graphics/IOFramebuffer.h>
#include <IOKit/graphics/IODisplay.h>

class IODisplayWrangler : public IOService
{
    OSDeclareDefaultStructors( IODisplayWrangler );

private:
    bool        fOpen;

    // from control panel: number of idle minutes before going off
    UInt32      fMinutesToDim;
    // false: use minutesToDim unless in emergency situation
    bool        fDimCaptured;
    bool        fAnnoyed;

    unsigned long fPendingPowerState;

    // ignore activity until time
    AbsoluteTime  fIdleUntil;
    // annoyed wake until time
    AbsoluteTime  fAnnoyanceUntil;

    AbsoluteTime  fDimInterval;
    AbsoluteTime  fSettingsChanged;
    AbsoluteTime  fOffInterval[2];

    AbsoluteTime  fPowerStateChangeTime;

    
    // IOService overrides
private:
    virtual IOReturn setAggressiveness( unsigned long, unsigned long ) APPLE_KEXT_OVERRIDE;
    virtual bool activityTickle( unsigned long, unsigned long ) APPLE_KEXT_OVERRIDE;
    virtual IOReturn setPowerState( unsigned long powerStateOrdinal, IOService* whatDevice ) APPLE_KEXT_OVERRIDE;
    virtual unsigned long initialPowerStateForDomainState( IOPMPowerFlags domainState ) APPLE_KEXT_OVERRIDE;

public:
    virtual bool start(IOService * provider) APPLE_KEXT_OVERRIDE;
    // Adaptive Dimming method
    virtual SInt32 nextIdleTimeout(AbsoluteTime currentTime,
                                   AbsoluteTime lastActivity, unsigned int powerState) APPLE_KEXT_OVERRIDE;

    // IORegistryEntry overrides
public:
    virtual OSObject * copyProperty( const char * aKey) const APPLE_KEXT_OVERRIDE;
    virtual IOReturn setProperties( OSObject * properties ) APPLE_KEXT_OVERRIDE;


private:
    virtual void initForPM( void );

    virtual IODisplayConnect * getDisplayConnect(
                IOFramebuffer * fb, IOIndex connect );

    virtual IOReturn getConnectFlagsForDisplayMode(
                IODisplayConnect * connect,
                IODisplayModeID mode, UInt32 * flags );

public:
    
    static bool serverStart(void);

    static bool makeDisplayConnects( IOFramebuffer * fb );
    static void destroyDisplayConnects( IOFramebuffer * fb );
    static void activityChange( IOFramebuffer * fb );
    static unsigned long getDisplaysPowerState(void);

    static IOReturn getFlagsForDisplayMode(
                IOFramebuffer * fb,
                IODisplayModeID mode, UInt32 * flags );

    void forceIdleImpl();
    static void forceIdle();

    static void builtinPanelPowerNotify(bool state);

};

void IODisplayUpdateNVRAM( IOService * entry, OSData * property );

#endif /* _IOKIT_IODISPLAYWRANGLER_H */
