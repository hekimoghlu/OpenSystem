/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 15, 2023.
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
#if PyObjC_BUILD_RELEASE >= 1006
    p = PyObjC_IdToPython(@protocol(NSAlertDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSAnimatablePropertyContainer)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSAnimationDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSApplicationDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSBrowserDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSChangeSpelling)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSCollectionViewDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSColorPickingCustom)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSColorPickingDefault)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSComboBoxCellDataSource)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSComboBoxDataSource)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSComboBoxDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSControlTextEditingDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSDatePickerCellDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSDockTilePlugIn)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSDraggingInfo)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSDrawerDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSGlyphStorage)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSIgnoreMisspelledWords)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSImageDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSInputServerMouseTracker)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSInputServiceProvider)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSLayoutManagerDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSMatrixDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSMenuDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSOpenSavePanelDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSOutlineViewDataSource)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSOutlineViewDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSPasteboardItemDataProvider)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSPasteboardReading)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSPasteboardWriting)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSPathCellDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSPathControlDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSPrintPanelAccessorizing)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSRuleEditorDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSSoundDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSSpeechRecognizerDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSSpeechSynthesizerDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSSplitViewDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSTabViewDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSTableViewDataSource)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSTableViewDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSTextAttachmentCell)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSTextDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSTextFieldDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSTextInput)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSTextInputClient)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSTextStorageDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSTextViewDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSTokenFieldCellDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSTokenFieldDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSToolbarDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSUserInterfaceItemSearching)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSUserInterfaceValidations)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSValidatedUserInterfaceItem)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSWindowDelegate)); Py_XDECREF(p);
#endif /* PyObjC_BUILD_RELEASE >= 1006 */
#if PyObjC_BUILD_RELEASE >= 1007
    p = PyObjC_IdToPython(@protocol(NSDraggingDestination)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSDraggingSource)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSPopoverDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSTextFinderBarContainer)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSTextFinderClient)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSTextLayoutOrientationProvider)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSUserInterfaceItemIdentification)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSWindowRestoration)); Py_XDECREF(p);
#endif /* PyObjC_BUILD_RELEASE >= 1007 */
#if PyObjC_BUILD_RELEASE >= 1008
    p = PyObjC_IdToPython(@protocol(NSPageControllerDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSSharingServiceDelegate)); Py_XDECREF(p);
    p = PyObjC_IdToPython(@protocol(NSSharingServicePickerDelegate)); Py_XDECREF(p);
#endif /* PyObjC_BUILD_RELEASE >= 1008 */
}
