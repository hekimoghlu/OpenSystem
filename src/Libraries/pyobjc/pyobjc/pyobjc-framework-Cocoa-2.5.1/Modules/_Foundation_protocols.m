/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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
static void __attribute__((__used__)) use_protocols(void)
{
    PyObject* p;
#if PyObjC_BUILD_RELEASE >= 1005
    p = PyObjC_IdToPython(@protocol(NSCoding)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSCopying)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSDecimalNumberBehaviors)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSFastEnumeration)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSLocking)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSMutableCopying)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSObject)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSURLAuthenticationChallengeSender)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSURLHandleClient)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSURLProtocolClient)); Py_XDECREF(p);
#endif /* PyObjC_BUILD_RELEASE >= 1005 */
#if PyObjC_BUILD_RELEASE >= 1006
    p = PyObjC_IdToPython(@protocol(NSCacheDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSConnectionDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSDiscardableContent)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSKeyedArchiverDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSKeyedUnarchiverDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSMachPortDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSMetadataQueryDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSNetServiceBrowserDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSNetServiceDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSPortDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSSpellServerDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSStreamDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSXMLParserDelegate)); Py_XDECREF(p);
#endif /* PyObjC_BUILD_RELEASE >= 1006 */
#if PyObjC_BUILD_RELEASE >= 1007
    p = PyObjC_IdToPython(@protocol(NSFileManagerDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSFilePresenter)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSURLConnectionDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSURLDownloadDelegate)); Py_XDECREF(p);
#endif /* PyObjC_BUILD_RELEASE >= 1007 */
#if PyObjC_BUILD_RELEASE >= 1008
    p = PyObjC_IdToPython(@protocol(NSSecureCoding)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSURLConnectionDataDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSURLConnectionDownloadDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSUserNotificationCenterDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSXPCListenerDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSXPCProxyCreating)); Py_XDECREF(p);
#endif /* PyObjC_BUILD_RELEASE >= 1008 */
}
