/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 29, 2025.
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
    p = PyObjC_IdToPython(@protocol(DOMEventListener)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(DOMEventTarget)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(DOMNodeFilter)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(DOMXPathNSResolver)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(WebDocumentRepresentation)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(WebDocumentSearching)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(WebDocumentText)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(WebDocumentView)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(WebOpenPanelResultListener)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(WebPlugInViewFactory)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(WebPolicyDecisionListener)); Py_XDECREF(p);
#endif /* PyObjC_BUILD_RELEASE >= 1005 */
}
