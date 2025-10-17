/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 18, 2021.
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
#include <libkern/c++/OSCollectionIterator.h>
#include <IOKit/assert.h>
#include <IOKit/IOKitKeys.h>
#include <IOKit/IOLib.h>
#include <IOKit/IOService.h>
#include <IOKit/hidsystem/IOHIDevice.h>
#include <IOKit/hidsystem/IOHIDParameter.h>
#include "IOHIDevicePrivateKeys.h"
#include "IOHIDFamilyPrivate.h"
#include "IOHIDEventService.h"

#define super IOService
OSDefineMetaClassAndStructors(IOHIDevice, IOService);

bool IOHIDevice::init(OSDictionary * properties)
{
  if (!super::init(properties))  return false;

  /*
   * Initialize minimal state.
   */

  return true;
}

void IOHIDevice::free()
{
  super::free();
}

bool IOHIDevice::start(IOService * provider) 
{
    if (!super::start(provider))
        return false;
   
    // RY: If the kIOHIDVirtualHIDevice property isn't
    // set scan the up provider chain to determine if
    // this is a resource or if the provider is another
    // IOHIDevice.  Also propegate value is property
    // was already set in provider.
    if (!getProperty(kIOHIDVirtualHIDevice)) {
        OSObject * prop = NULL;

        while (provider) {
            prop = provider->copyProperty(kIOHIDVirtualHIDevice);
            if ( OSDynamicCast(OSBoolean, prop) ) {
                setProperty(kIOHIDVirtualHIDevice, prop);
                break;
            }
            else if ( provider == getResourceService() || OSDynamicCast(IOHIDevice, provider) ) {
                setProperty(kIOHIDVirtualHIDevice, kOSBooleanTrue);
                break;
            }

            provider = provider->getProvider();
            OSSafeReleaseNULL(prop);
        }
        OSSafeReleaseNULL(prop);
        
        if ( !provider )
            setProperty(kIOHIDVirtualHIDevice, kOSBooleanFalse);
    }
    
    updateProperties();
    
    return true;
}

bool IOHIDevice::open(IOService *    forClient,
                      IOOptionBits   options,
                      void *         arg)
{
    if(forClient == this) return true;
    
    return super::open(forClient, options, arg);
}


IOHIDKind IOHIDevice::hidKind()
{
  return kHIUnknownDevice;
}

UInt32 IOHIDevice::interfaceID()
{
  return 0;
}

UInt32 IOHIDevice::deviceType()
{
  return 0;
}

UInt64 IOHIDevice::getGUID()
{
  return(0xffffffffffffffffULL);
}

SInt32 IOHIDevice::GenerateKey(OSObject *object)
{
    if ( !object )
        return 0;
    IORegistryEntry* temp = (IORegistryEntry*)(object);
    return (SInt32)temp->getRegistryEntryID(); // Get unique key for this object;
}

bool IOHIDevice::updateProperties( void )
{
    bool ok;

    ok = setProperty( kIOHIDKindKey, hidKind(), 32 ) &&
         setProperty( kIOHIDInterfaceIDKey, interfaceID(), 32 ) &&
         setProperty( kIOHIDDeviceEventIDKey, IOHIDevice::GenerateKey(this), 32 ) &&
         setProperty( kIOHIDSubinterfaceIDKey, deviceType(), 32 );

    return( ok );
}

// RY: Override IORegistryEntry::setProperties().  This will allow properties
// to be set per device, instead of globally via setParamProperties.
IOReturn IOHIDevice::setProperties( OSObject * properties )
{
    OSDictionary * propertyDict = OSDynamicCast(OSDictionary, properties);
    IOReturn       ret          = kIOReturnBadArgument;

    if ( propertyDict ) {
        if (propertyDict->setOptions(0, 0) & OSDictionary::kImmutable) {
            OSDictionary * temp = propertyDict;
            propertyDict = OSDynamicCast(OSDictionary, temp->copyCollection());
        }
        else {
            propertyDict->retain();
        }
        propertyDict->setObject(kIOHIDDeviceParametersKey, kOSBooleanTrue);
        ret = setParamProperties( propertyDict );
        propertyDict->removeObject(kIOHIDDeviceParametersKey);
        propertyDict->release();
    }

    return ret;
}


IOReturn IOHIDevice::setParamProperties( OSDictionary * dict )
{
    IOHIDEventService * eventService = NULL;
    
    if ( dict->getObject(kIOHIDEventServicePropertiesKey) == NULL ) {
        IOService * service = getProvider();
        if ( service )
            eventService = OSDynamicCast(IOHIDEventService, service);
    }

    if ( dict->getObject(kIOHIDDeviceParametersKey) == kOSBooleanTrue ) {
        OSDictionary * deviceParameters = OSDynamicCast(OSDictionary, copyProperty(kIOHIDParametersKey));

        if ( !deviceParameters ) {
            deviceParameters = OSDictionary::withCapacity(4);
        }
        else {
            // each writer must have seprate copy to modify property
            // to avoid race condition and call to global copy is
            // handled through sync lock
            OSDictionary * temp = deviceParameters;
            deviceParameters = OSDynamicCast(OSDictionary, temp->copyCollection());
            temp->release();
        }

        if ( deviceParameters ) {
            // RY: Because K&M Prefs and Admin still expect device props to be
            // top level, let's continue to set them via setProperty. When we get
            // Max to migrate over, we can remove the interator code and use:
            // deviceParameters->merge(dict);
            // deviceParameters->removeObject(kIOHIDResetKeyboardKey);
            // deviceParameters->removeObject(kIOHIDResetPointerKey);
            // setProperty(kIOHIDParametersKey, deviceParameters);
            // deviceParameters->release();

            OSCollectionIterator * iterator = OSCollectionIterator::withCollection(dict);
            if ( iterator ) {
                OSSymbol * key;

                while ( ( key = (OSSymbol *)iterator->getNextObject() ) )
                    if (    !key->isEqualTo(kIOHIDResetKeyboardKey) && 
                            !key->isEqualTo(kIOHIDResetPointerKey) && 
                            !key->isEqualTo(kIOHIDScrollResetKey) && 
                            !key->isEqualTo(kIOHIDDeviceParametersKey) && 
                            !key->isEqualTo(kIOHIDResetLEDsKey) &&
                            !IsIOHIDRestrictedIOKitProperty(key)) {
                        OSObject * value = dict->getObject(key);

                        deviceParameters->setObject(key, value);
                        setProperty(key, value);
                    }

                iterator->release();
            }

            setProperty(kIOHIDParametersKey, deviceParameters);
            deviceParameters->release();

            // RY: Propogate up to IOHIDEventService level
            if ( eventService )
                eventService->setSystemProperties(dict);

        }
        else {
            return kIOReturnNoMemory;
        }
    }

    return( kIOReturnSuccess );
}


