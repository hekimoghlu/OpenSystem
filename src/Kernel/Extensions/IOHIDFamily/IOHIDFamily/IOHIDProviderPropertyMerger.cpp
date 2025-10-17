/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 26, 2024.
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
#include <AssertMacros.h>
#include <libkern/OSAtomic.h>
#include <libkern/c++/OSDictionary.h>
#include <IOKit/IOLib.h>
#include <IOKit/IOProviderPropertyMerger.h>

#include "IOHIDProviderPropertyMerger.h"

#define super IOProviderPropertyMerger

OSDefineMetaClassAndStructors(IOHIDProviderPropertyMerger, IOProviderPropertyMerger);

IOService * IOHIDProviderPropertyMerger::probe(IOService *provider, SInt32 * score __unused)
{
    OSObject * properties = NULL;

    properties = copyProperty(kIOProviderMergePropertiesKey);
    mergeProperties(provider, OSDynamicCast(OSDictionary, properties));
    OSSafeReleaseNULL(properties);

    // Fail probe by returning no service.
    return NULL;
}

bool IOHIDProviderPropertyMerger::mergeProperties(IOService * provider, OSDictionary * properties)
{
    const OSSymbol *        dictionaryEntry = NULL;
    OSCollectionIterator *  iterator        = NULL;
    bool                    result          = false;

    require(provider && properties, exit);

    // Iterate through the properties until we run out of entries
    iterator = OSCollectionIterator::withCollection(properties);
    require(iterator, exit);

    while ( (dictionaryEntry = (const OSSymbol *)iterator->getNextObject()) ) {
        OSDictionary *	sourceDictionary    = NULL;
        OSObject *      providerObject      = NULL;
        OSDictionary *	providerDictionary  = NULL;
        
        providerObject = provider->copyProperty(dictionaryEntry);
        
        // See if our source entry is also a dictionary
        sourceDictionary    = OSDynamicCast(OSDictionary, properties->getObject(dictionaryEntry));
        providerDictionary  = OSDynamicCast(OSDictionary, providerObject);
        
        if ( providerDictionary && sourceDictionary )  {

            // Because access to the registry table may not be synchronized, we should take a copy
            OSDictionary *  providerDictionaryCopy = NULL;

            providerDictionaryCopy = OSDictionary::withDictionary( providerDictionary, 0);
            require_action(providerDictionaryCopy, dictionaryExit, result=false);
            
            // Recursively merge the two dictionaries
            result = mergeDictionaries(sourceDictionary, providerDictionaryCopy);
            require(result, dictionaryExit);
            
            // OK, now we can just set the property in our provider
            result = provider->setProperty(dictionaryEntry, providerDictionaryCopy);
            require(result, dictionaryExit);

dictionaryExit:
            if ( providerDictionaryCopy )
                providerDictionaryCopy->release();
        } else {
            // Not a dictionary, so just set the property if it is not present on provider
            if (!providerObject) {
                result = provider->setProperty(dictionaryEntry, properties->getObject(dictionaryEntry));
            }
        }
        
        if ( providerObject )
            providerObject->release();
        
        if ( !result )
            break;
    }

exit:
    if ( iterator )
        iterator->release();

    return result;
}


bool IOHIDProviderPropertyMerger::mergeDictionaries(OSDictionary * source,  OSDictionary * target)
{
    OSCollectionIterator *  srcIterator = NULL;
    OSSymbol*               keyObject   = NULL;
    bool                    result      = false;

    require(source && target, exit);

    // Get our source dictionary
    srcIterator = OSCollectionIterator::withCollection(source);
    require(srcIterator, exit);

    while ((keyObject = OSDynamicCast(OSSymbol, srcIterator->getNextObject()))) {
        OSDictionary *	childSourceDictionary   = NULL;
        OSDictionary *	childTargetDictionary   = NULL;
        OSObject *      childTargetObject       = NULL;

        // Check to see if our destination already has the same entry.
        childTargetObject = target->getObject(keyObject);
        if ( childTargetObject )
            childTargetDictionary = OSDynamicCast(OSDictionary, childTargetObject);

        // See if our source entry is also a dictionary
        childSourceDictionary = OSDynamicCast(OSDictionary, source->getObject(keyObject));

        if ( childTargetDictionary && childSourceDictionary) {
            // Our destination dictionary already has the entry for this same object AND our
            // source is also a dcitionary, so we need to recursively add it.
            //
            result = mergeDictionaries(childSourceDictionary, childTargetDictionary) ;
            if ( !result )
                break;
        } else {
            // We have a property that we need to merge into our parent dictionary.
            //
            result = target->setObject(keyObject, source->getObject(keyObject)) ;
            if ( !result )
                break;
        }
    }

exit:
    if ( srcIterator )
        srcIterator->release();

    return result;
}
