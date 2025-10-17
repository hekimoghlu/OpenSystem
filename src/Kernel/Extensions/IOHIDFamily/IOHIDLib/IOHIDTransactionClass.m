/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 17, 2021.
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
#import <IOKit/hid/IOHIDLib.h>
#import "IOHIDTransactionClass.h"
#import "HIDLibElement.h"
#import <AssertMacros.h>
#import "IOHIDLibUserClient.h"
#import <IOKit/hid/IOHIDLibPrivate.h>
#import <os/assumes.h>

@implementation IOHIDTransactionClass

- (HRESULT)queryInterface:(REFIID)uuidBytes
             outInterface:(LPVOID *)outInterface
{
    CFUUIDRef uuid = CFUUIDCreateFromUUIDBytes(NULL, uuidBytes);
    HRESULT result = E_NOINTERFACE;
    
    if (CFEqual(uuid, kIOHIDDeviceTransactionInterfaceID)) {
        *outInterface = (LPVOID *)&_interface;
        CFRetain((CFTypeRef)self);
        result = S_OK;
    }
    
    if (uuid) {
        CFRelease(uuid);
    }
    
    return result;
}

static IOReturn _getAsyncEventSource(void *iunknown, CFTypeRef *pSource)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDTransactionClass *me = (__bridge id)vtbl->_reserved;
    
    return [me getAsyncEventSource:pSource];
}

- (IOReturn)getAsyncEventSource:(CFTypeRef *)pSource
{
    if (!pSource) {
        return kIOReturnBadArgument;
    }
    
    *pSource = _device.runLoopSource;
    
    return kIOReturnSuccess;
}

static IOReturn _setDirection(void *iunknown,
                              IOHIDTransactionDirectionType direction,
                              IOOptionBits options __unused)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDTransactionClass *me = (__bridge id)vtbl->_reserved;
    
    return [me setDirection:direction];
}

- (IOReturn)setDirection:(IOHIDTransactionDirectionType)direction
{
    _direction = direction;
    return kIOReturnSuccess;
}

static IOReturn _getDirection(void *iunknown,
                              IOHIDTransactionDirectionType *pDirection)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDTransactionClass *me = (__bridge id)vtbl->_reserved;
    
    return [me getDirection:pDirection];
}

- (IOReturn)getDirection:(IOHIDTransactionDirectionType *)pDirection
{
    if (!pDirection) {
        return kIOReturnBadArgument;
    }
    
    *pDirection = _direction;
    return kIOReturnSuccess;
}

static IOReturn _addElement(void *iunknown,
                            IOHIDElementRef element,
                            IOOptionBits options __unused)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDTransactionClass *me = (__bridge id)vtbl->_reserved;
    
    return [me addElement:element];
}

- (IOReturn)addElement:(IOHIDElementRef)elementRef
{
    IOReturn ret = kIOReturnError;
    HIDLibElement *element = nil;
    
    require_action(elementRef, exit, ret = kIOReturnBadArgument);
    
    element = [[HIDLibElement alloc] initWithElementRef:elementRef];
    require(element, exit);
    
    require(![_elements containsObject:element], exit);
    
    if (_direction == kIOHIDTransactionDirectionTypeOutput) {
        require(element.type == kIOHIDElementTypeOutput ||
                element.type == kIOHIDElementTypeFeature, exit);
    }
    
    [_elements addObject:element];
    ret = kIOReturnSuccess;
    
exit:
    return ret;
}

static IOReturn _removeElement(void *iunknown,
                               IOHIDElementRef element,
                               IOOptionBits options __unused)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDTransactionClass *me = (__bridge id)vtbl->_reserved;
    
    return [me removeElement:element];
}

- (IOReturn)removeElement:(IOHIDElementRef)elementRef
{
    IOReturn ret = kIOReturnError;
    HIDLibElement *element = nil;
    
    require_action(elementRef, exit, ret = kIOReturnBadArgument);
    
    element = [[HIDLibElement alloc] initWithElementRef:elementRef];
    require(element && [_elements containsObject:element], exit);
    
    [_elements removeObject:element];
    ret = kIOReturnSuccess;
    
exit:
    return ret;
}

static IOReturn _containsElement(void *iunknown,
                                 IOHIDElementRef element,
                                 Boolean *pValue,
                                 IOOptionBits options __unused)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDTransactionClass *me = (__bridge id)vtbl->_reserved;
    
    return [me containsElement:element value:pValue];
}

- (IOReturn)containsElement:(IOHIDElementRef)elementRef
                      value:(Boolean *)pValue
{
    IOReturn ret = kIOReturnError;
    HIDLibElement *element = nil;
    
    require_action(elementRef && pValue, exit, ret = kIOReturnBadArgument);
    
    element = [[HIDLibElement alloc] initWithElementRef:elementRef];
    require(element, exit);
    
    *pValue = [_elements containsObject:element];
    ret = kIOReturnSuccess;
    
exit:
    return ret;
}

static IOReturn _setValue(void *iunknown,
                          IOHIDElementRef element,
                          IOHIDValueRef value,
                          IOOptionBits options)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDTransactionClass *me = (__bridge id)vtbl->_reserved;

    return [me setValue:element value:value options:options];
}

- (IOReturn)setValue:(IOHIDElementRef)elementRef
               value:(IOHIDValueRef)valueRef
             options:(IOOptionBits)options
{
    IOReturn ret = kIOReturnError;
    HIDLibElement *element = nil;
    
    require_action(elementRef && valueRef, exit, ret = kIOReturnBadArgument);
    require(_direction == kIOHIDTransactionDirectionTypeOutput, exit);
    
    element = [[HIDLibElement alloc] initWithElementRef:elementRef];
    require(element && [_elements containsObject:element], exit);
    
    if (options & kIOHIDTransactionOptionDefaultOutputValue) {
        [element setDefaultValueRef:valueRef];
    } else {
        [element setValueRef:valueRef];
    }
    
    [_elements replaceObjectAtIndex:[_elements indexOfObject:element]
                           withObject:element];
    
    ret = kIOReturnSuccess;
    
exit:
    return ret;
}

static IOReturn _getValue(void *iunknown,
                          IOHIDElementRef elementRef,
                          IOHIDValueRef *pValueRef,
                          IOOptionBits options)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDTransactionClass *me = (__bridge id)vtbl->_reserved;
    
    return [me getValue:elementRef value:pValueRef options:options];
}

- (IOReturn)getValue:(IOHIDElementRef)elementRef
               value:(IOHIDValueRef *)pValueRef
             options:(IOOptionBits)options
{
    IOReturn ret = kIOReturnError;
    HIDLibElement *element = nil;
    HIDLibElement *tmp = nil;
    
    require_action(elementRef && pValueRef, exit, ret = kIOReturnBadArgument);
    
    tmp = [[HIDLibElement alloc] initWithElementRef:elementRef];
    require(tmp && [_elements containsObject:tmp], exit);
    
    element = [_elements objectAtIndex:[_elements indexOfObject:tmp]];
    
    if (options & kIOHIDTransactionOptionDefaultOutputValue) {
        *pValueRef = element.defaultValueRef;
    } else {
        *pValueRef = element.valueRef;
    }

    ret = kIOReturnSuccess;
    
exit:
    return ret;
}

typedef struct {
    IOHIDCallback    callback;
    void           * context;
    void           * device;
    void           * transaction;
    NSArray        * elements;
} AsyncCommitContext;

static void _asyncCallback(void * context, IOReturn result, uint32_t bufferSize, uint64_t addr)
{
    HIDLibElement      * element;
    IOHIDValueRef        value;
    IOHIDElementValue  * elementVal;
    size_t               dataOffset   = 0;
    AsyncCommitContext * asyncContext = (AsyncCommitContext *)context;

    if (!asyncContext || !asyncContext->callback) {
        return;
    }

    if (asyncContext->elements && addr && bufferSize) {
        for (element in asyncContext->elements) {
            elementVal = (IOHIDElementValue *)((uint8_t *)addr + dataOffset);

            dataOffset += elementVal->totalSize;
            if (elementVal->totalSize < ELEMENT_VALUE_HEADER_SIZE(elementVal) || dataOffset > bufferSize) {
                HIDLogError("Unable to copy back value for element, unexpected size(%d)", elementVal->totalSize);
                break;
            } else if (elementVal->cookie != element.elementCookie) {
                HIDLogError("Unable to copy back value for element, unexpected cookie(%ld) expected:%d", (long)elementVal->cookie, element.elementCookie);
                break;
            }

            value = _IOHIDValueCreateWithElementValuePtr(kCFAllocatorDefault, element.elementRef, elementVal);
            [element setValueRef:value];
            if (value) {
                CFRelease(value);
            }
        }
        [(__bridge IOHIDDeviceClass *)asyncContext->device releaseReport:addr];
    }

    ((IOHIDCallback)asyncContext->callback)(asyncContext->context, result, asyncContext->transaction);

    asyncContext->elements = NULL;
    free(asyncContext);
}

static IOReturn _commit(void *iunknown,
                        uint32_t timeout,
                        IOHIDCallback callback,
                        void *context,
                        IOOptionBits options)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDTransactionClass *me = (__bridge id)vtbl->_reserved;
    
    return [me commit:context timeout:timeout callback:callback options:options];
}

- (IOReturn)commit:(void *)context
           timeout:(uint32_t)timeout
          callback:(IOHIDCallback)callback
           options:(IOOptionBits)options
{
    uint64_t                  regID;
    IOReturn                  ret          = kIOReturnError;
    uint64_t                  input[3]     = {0};
    size_t                    dataSize     = 0;
    size_t                    dataOffset   = 0;
    void                    * cookies      = NULL;
    void                    * elementData  = NULL;
    uint32_t                  count        = (uint32_t)_elements.count;
    size_t                    cookiesSize  = count * sizeof(uint32_t);
    AsyncCommitContext      * asyncContext = NULL;
    HIDLibElement           * element;
    size_t                    elementSize;
    IOHIDValueRef             value;
    IOHIDElementValue       * elementVal;
    IOHIDElementValueHeader * elementValHeader;
    io_async_ref64_t          asyncRef;

    require(count, exit);

    IORegistryEntryGetRegistryEntryID(_device.service, &regID);

    input[2] = options;

    if (callback) {
        input[0] = timeout;
        require((asyncContext = (AsyncCommitContext *)calloc(1, sizeof(AsyncCommitContext))), exit);

        asyncContext->callback    = callback;
        asyncContext->context     = context;
        asyncContext->device      = (__bridge void *)_device;
        asyncContext->transaction = &_interface;
        asyncContext->elements    = NULL;

        asyncRef[kIOAsyncCalloutFuncIndex] = (uint64_t)_asyncCallback;
        asyncRef[kIOAsyncCalloutRefconIndex] = (uint64_t)asyncContext;
    }

    if (_direction == kIOHIDTransactionDirectionTypeOutput) {
        for (uint32_t i = 0; i < count; i++) {
            element = [_elements objectAtIndex:i];

            ret = [_device setValue:element.elementRef value:element.valueRef timeout:0 callback:nil context:nil options:kHIDSetElementValuePendEvent];
            require_noerr_action(ret, exit, HIDLogError("setValue(%#llx):%#x", regID, ret));
            
            elementSize = sizeof(IOHIDElementValueHeader) + IOHIDValueGetLength(element.valueRef);
            dataSize += elementSize;
        }

        require_action((elementData = malloc(dataSize)), exit, ret = kIOReturnNoMemory);

        for (uint32_t i = 0; i < count; i++) {
            element = [_elements objectAtIndex:i];
            elementValHeader = (IOHIDElementValueHeader *)(elementData + dataOffset);
            _IOHIDValueCopyToElementValueHeader(element.valueRef, elementValHeader);
            dataOffset += elementValHeader->length + sizeof(IOHIDElementValueHeader);
        }

        if (callback) {
            ret = IOConnectCallAsyncMethod(_device.connect, kIOHIDLibUserClientPostElementValues, [_device getPort], asyncRef, kIOAsyncCalloutCount, input, 1, elementData, dataSize, NULL, NULL, NULL, NULL);
        } else {
            ret = IOConnectCallMethod(_device.connect, kIOHIDLibUserClientPostElementValues, input, 1, elementData, dataSize, NULL, NULL, NULL, NULL);
        }
        require_noerr_action(ret, exit, HIDLogError("kIOHIDLibUserClientPostElementValues(%#llx):%#x", regID, ret));

    } else {
        for (uint32_t i = 0; i < count; i++) {
            element = [_elements objectAtIndex:i];
            elementSize = sizeof(IOHIDElementValue) + _IOHIDElementGetLength(element.elementRef);
            value = element.valueRef;
            dataSize += elementSize;

            ret = [_device getValue:element.elementRef value:&value timeout:0 callback:nil context:nil options:kHIDGetElementValuePendEvent];
            require_noerr_action(ret, exit, HIDLogError("getValue(%#llx):%#x", regID, ret));
        }

        require_action((cookies = malloc(cookiesSize)), exit, ret = kIOReturnNoMemory);

        for (uint32_t i = 0; i < count; i++) {
            element = [_elements objectAtIndex:i];
            *((uint32_t *)cookies + i) = (uint32_t)element.elementCookie;
        }

        if (callback) {
            input[1] = dataSize;
            require_action((asyncContext->elements = [NSArray arrayWithArray:_elements]), exit, ret = kIOReturnNoMemory);
            ret = IOConnectCallAsyncMethod(_device.connect, kIOHIDLibUserClientUpdateElementValues, [_device getPort], asyncRef, kIOAsyncCalloutCount, input, 3, cookies, cookiesSize, NULL, NULL, NULL, NULL);
            require_noerr_action(ret, exit, HIDLogError("kIOHIDLibUserClientUpdateElementValues(%#llx):%#x", regID, ret));
        } else {
            require_action((elementData = calloc(1, dataSize)), exit, ret = kIOReturnNoMemory);
            ret = IOConnectCallMethod(_device.connect, kIOHIDLibUserClientUpdateElementValues, input, 3, cookies, cookiesSize, NULL, NULL, elementData, &dataSize);
            require_noerr_action(ret, exit, HIDLogError("kIOHIDLibUserClientUpdateElementValues(%#llx):%#x", regID, ret));

            for (element in _elements) {
                elementVal = (IOHIDElementValue *)((uint8_t *)elementData + dataOffset);
                dataOffset += elementVal->totalSize;

                if (elementVal->totalSize < ELEMENT_VALUE_HEADER_SIZE(elementVal) || dataOffset > dataSize) {
                    HIDLogError("Unable to copy back value for element, unexpected size(%d)", elementVal->totalSize);
                    break;
                } else if (elementVal->cookie != element.elementCookie) {
                    HIDLogError("Unable to copy back value for element, unexpected cookie(%ld) expected:%d", (long)elementVal->cookie, element.elementCookie);
                    break;
                }

                value = _IOHIDValueCreateWithElementValuePtr(kCFAllocatorDefault, element.elementRef, elementVal);
                [element setValueRef:value];
                if (value) {
                    CFRelease(value);
                }
            }
        }
    }

exit:
    if (cookies) {
        free(cookies);
    }
    if (elementData) {
        free(elementData);
    }
    if (asyncContext && ret) {
        asyncContext->elements = NULL;
        free(asyncContext);
    }

    return ret;
}

static IOReturn _clear(void *iunknown, IOOptionBits options __unused)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDTransactionClass *me = (__bridge id)vtbl->_reserved;
    
    return [me clear];
}

- (IOReturn)clear
{
    [_elements removeAllObjects];
    return kIOReturnSuccess;
}

- (instancetype)initWithDevice:(IOHIDDeviceClass *)device
{
    self = [super init];
    
    if (!self) {
        return nil;
    }
    
    _device = device;
    
    _interface = (IOHIDDeviceTransactionInterface *)malloc(sizeof(*_interface));
    
    *_interface = (IOHIDDeviceTransactionInterface) {
        // IUNKNOWN_C_GUTS
        ._reserved = (__bridge void *)self,
        .QueryInterface = self->_vtbl->QueryInterface,
        .AddRef = self->_vtbl->AddRef,
        .Release = self->_vtbl->Release,
        
        // IOHIDDeviceTransactionInterface
        .getAsyncEventSource = _getAsyncEventSource,
        .setDirection = _setDirection,
        .getDirection = _getDirection,
        .addElement = _addElement,
        .removeElement = _removeElement,
        .containsElement = _containsElement,
        .setValue = _setValue,
        .getValue = _getValue,
        .commit = _commit,
        .clear = _clear,
    };
    
    _elements = [[NSMutableArray alloc] init];
    
    return self;
}

- (void)setDevice:(IOHIDDeviceClass *)device
{
    _device = device;
}

- (IOHIDDeviceClass *)device
{
    return _device;
}

- (void)dealloc
{
    free(_interface);
}

@end

@implementation IOHIDOutputTransactionClass

- (HRESULT)queryInterface:(REFIID)uuidBytes
             outInterface:(LPVOID *)outInterface
{
    CFUUIDRef uuid = CFUUIDCreateFromUUIDBytes(NULL, uuidBytes);
    HRESULT result = E_NOINTERFACE;
    
    if (CFEqual(uuid, kIOHIDOutputTransactionInterfaceID)) {
        *outInterface = (LPVOID *)&_outputInterface;
        CFRetain((CFTypeRef)self);
        result = S_OK;
    }
    
    if (uuid) {
        CFRelease(uuid);
    }
    
    return result;
}

static IOReturn _createAsyncEventSource(void *iunknown,
                                        CFRunLoopSourceRef *source)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDOutputTransactionClass *me = (__bridge id)vtbl->_reserved;
    
    if (!source) {
        return kIOReturnBadArgument;
    }
    
    *source = me->_device.runLoopSource;
    CFRetain(*source);
    
    return kIOReturnSuccess;
}

static CFRunLoopSourceRef _getOutputAsyncEventSource(void *iunknown)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDOutputTransactionClass *me = (__bridge id)vtbl->_reserved;
    
    return me->_device.runLoopSource;
}

static IOReturn _createAsyncPort(void *iunknown, mach_port_t *port)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDOutputTransactionClass *me = (__bridge id)vtbl->_reserved;
    
    *port = me->_device.port;
    return kIOReturnSuccess;
}

static mach_port_t _getAsyncPort(void *iunknown)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDOutputTransactionClass *me = (__bridge id)vtbl->_reserved;
    
    return me->_device.port;
}

static IOReturn _create(void *iunknown __unused)
{
    return kIOReturnSuccess;
}

static IOReturn _dispose(void *iunknown __unused)
{
    return kIOReturnSuccess;
}

static IOReturn _addOutputElement(void *iunknown,
                                  IOHIDElementCookie elementCookie)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDOutputTransactionClass *me = (__bridge id)vtbl->_reserved;
    
    return [me addElement:[me->_device getElement:(uint32_t)elementCookie]];
}

static IOReturn _removeOutputElement(void *iunknown,
                                     IOHIDElementCookie elementCookie)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDOutputTransactionClass *me = (__bridge id)vtbl->_reserved;
    
    return [me removeElement:[me->_device getElement:(uint32_t)elementCookie]];
}

static Boolean _hasElement(void *iunknown, IOHIDElementCookie elementCookie)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDOutputTransactionClass *me = (__bridge id)vtbl->_reserved;
    
    Boolean contains = false;
    
    [me containsElement:[me->_device getElement:(uint32_t)elementCookie]
                  value:&contains];
    
    return contains;
}

static IOReturn _setElementDefault(void *iunknown,
                                   IOHIDElementCookie elementCookie,
                                   IOHIDEventStruct *valueEvent)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDOutputTransactionClass *me = (__bridge id)vtbl->_reserved;
    
    return [me setElementValue:elementCookie
                         value:valueEvent
                       options:kIOHIDTransactionOptionDefaultOutputValue];
}

static IOReturn _getElementDefault(void *iunknown,
                                   IOHIDElementCookie elementCookie,
                                   IOHIDEventStruct *outValueEvent)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDOutputTransactionClass *me = (__bridge id)vtbl->_reserved;
    
    return [me getElementValue:elementCookie
                         value:outValueEvent
                       options:kIOHIDTransactionOptionDefaultOutputValue];
}

static IOReturn _setElementValue(void *iunknown,
                                 IOHIDElementCookie elementCookie,
                                 IOHIDEventStruct *valueEvent)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDOutputTransactionClass *me = (__bridge id)vtbl->_reserved;
    
    return [me setElementValue:elementCookie value:valueEvent options:0];
}

- (IOReturn)setElementValue:(IOHIDElementCookie)elementCookie
                      value:(IOHIDEventStruct *)eventStruct
                    options:(IOOptionBits)options
{
    if (!eventStruct) {
        return kIOReturnBadArgument;
    }
    
    IOReturn ret = kIOReturnError;
    IOHIDElementRef elementRef = [_device getElement:(uint32_t)elementCookie];
    IOHIDValueRef value = _IOHIDValueCreateWithStruct(kCFAllocatorDefault,
                                                      elementRef,
                                                      eventStruct);
    
    require(elementRef && value, exit);
    
    ret = [self setValue:elementRef value:value options:options];
    
exit:
    if (value) {
        CFRelease(value);
    }
    
    return ret;
}

static IOReturn _getElementValue(void *iunknown,
                                 IOHIDElementCookie elementCookie,
                                 IOHIDEventStruct *outValueEvent)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDOutputTransactionClass *me = (__bridge id)vtbl->_reserved;
    
    return [me getElementValue:elementCookie value:outValueEvent options:0];
}

- (IOReturn)getElementValue:(IOHIDElementCookie)elementCookie
                      value:(IOHIDEventStruct *)eventStruct
                    options:(IOOptionBits)options
{
    if (!eventStruct) {
        return kIOReturnBadArgument;
    }
    
    IOReturn ret = kIOReturnError;
    IOHIDElementRef elementRef = [_device getElement:(uint32_t)elementCookie];
    HIDLibElement *element;
    IOHIDValueRef value;
    uint32_t length;
    
    ret = [self getValue:elementRef value:&value options:options];
    require_noerr(ret, exit);
    
    elementRef = IOHIDValueGetElement(value);
    element = [[HIDLibElement alloc] initWithElementRef:elementRef];
    element.valueRef = value;
    
    length = (uint32_t)element.length;
    eventStruct->type = element.type;
    eventStruct->elementCookie = (IOHIDElementCookie)element.elementCookie;
    *(UInt64 *)&eventStruct->timestamp = element.timestamp;
    
    if (length > sizeof(uint32_t)) {
        eventStruct->longValueSize = length;
        eventStruct->longValue = malloc(length);
        bcopy(IOHIDValueGetBytePtr(value), eventStruct->longValue, length);
    } else {
        eventStruct->longValueSize = 0;
        eventStruct->longValue = NULL;
        eventStruct->value = (int32_t)element.integerValue;
    }
    
exit:
    return ret;
}

static IOReturn _commitOutput(void *iunknown,
                              uint32_t timeoutMS __unused,
                              IOHIDCallbackFunction callback __unused,
                              void *callbackTarget __unused,
                              void *callbackRefcon __unused)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDOutputTransactionClass *me = (__bridge id)vtbl->_reserved;
    
    return [me commit:nil timeout:0 callback:nil options:0];
}

static IOReturn _clearOutput(void *iunknown)
{
    IUnknownVTbl *vtbl = *((IUnknownVTbl**)iunknown);
    IOHIDOutputTransactionClass *me = (__bridge id)vtbl->_reserved;
    
    return [me clear];
}

- (instancetype)initWithDevice:(IOHIDDeviceClass *)device
{
    self = [super initWithDevice:device];
    
    if (!self) {
        return nil;
    }
    
    _direction = kIOHIDTransactionDirectionTypeOutput;
    
    _outputInterface = (IOHIDOutputTransactionInterface *)malloc(sizeof(*_outputInterface));
    
    *_outputInterface = (IOHIDOutputTransactionInterface) {
        // IUNKNOWN_C_GUTS
        ._reserved = (__bridge void *)self,
        .QueryInterface = self->_vtbl->QueryInterface,
        .AddRef = self->_vtbl->AddRef,
        .Release = self->_vtbl->Release,
        
        // IOHIDOutputTransactionInterface
        .createAsyncEventSource = _createAsyncEventSource,
        .getAsyncEventSource = _getOutputAsyncEventSource,
        .createAsyncPort = _createAsyncPort,
        .getAsyncPort = _getAsyncPort,
        .create = _create,
        .dispose = _dispose,
        .addElement = _addOutputElement,
        .removeElement = _removeOutputElement,
        .hasElement = _hasElement,
        .setElementDefault = _setElementDefault,
        .getElementDefault = _getElementDefault,
        .setElementValue = _setElementValue,
        .getElementValue = _getElementValue,
        .commit = _commitOutput,
        .clear = _clearOutput,
    };
    
    return self;
}

- (void)dealloc
{
    free(_outputInterface);
}

@end
