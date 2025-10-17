/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 23, 2024.
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
#pragma once

#include "BaseClickableWithKeyInputType.h"
#include "FileChooser.h"
#include "FileIconLoader.h"
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class DirectoryFileListCreator;
class DragData;
class FileList;
class Icon;

class FileInputType final : public BaseClickableWithKeyInputType, private FileChooserClient, private FileIconLoaderClient, public CanMakeWeakPtr<FileInputType> {
    WTF_MAKE_TZONE_ALLOCATED(FileInputType);
public:
    static Ref<FileInputType> create(HTMLInputElement& element)
    {
        return adoptRef(*new FileInputType(element));
    }

    virtual ~FileInputType();

    String firstElementPathForInputValue() const; // Checked first, before internal storage or the value attribute.
    FileList& files() { return m_fileList; }
    void setFiles(RefPtr<FileList>&&, WasSetByJavaScript);

    static std::pair<Vector<FileChooserFileInfo>, String> filesFromFormControlState(const FormControlState&);
    bool canSetStringValue() const final;
    bool valueMissing(const String&) const final;

private:
    explicit FileInputType(HTMLInputElement&);

    const AtomString& formControlType() const final;
    FormControlState saveFormControlState() const final;
    void restoreFormControlState(const FormControlState&) final;
    bool appendFormData(DOMFormData&) const final;
    String valueMissingText() const final;
    void handleDOMActivateEvent(Event&) final;
    RenderPtr<RenderElement> createInputRenderer(RenderStyle&&) final;
    enum class RequestIcon : bool { No, Yes };
    void setFiles(RefPtr<FileList>&&, RequestIcon, WasSetByJavaScript);
    String displayString() const final;
    void setValue(const String&, bool valueChanged, TextFieldEventBehavior, TextControlSetValueSelection) final;
    void showPicker() final;
    bool allowsShowPickerAcrossFrames() final;

#if ENABLE(DRAG_SUPPORT)
    bool receiveDroppedFilesWithImageTranscoding(const Vector<String>& paths);
    bool receiveDroppedFiles(const DragData&) final;
#endif

    Icon* icon() const final;
    void createShadowSubtree() final;
    void disabledStateChanged() final;
    void attributeChanged(const QualifiedName&) final;
    String defaultToolTip() const final;

    void filesChosen(const Vector<FileChooserFileInfo>&, const String& displayString = { }, Icon* = nullptr) final;
    void filesChosen(const Vector<String>& paths, const Vector<String>& replacementPaths = { });
    void fileChoosingCancelled();

    // FileIconLoaderClient implementation.
    void iconLoaded(RefPtr<Icon>&&) final;

    FileChooserSettings fileChooserSettings() const;
    void applyFileChooserSettings();
    void didCreateFileList(Ref<FileList>&&, RefPtr<Icon>&&);
    void requestIcon(const Vector<String>&);

    bool allowsDirectories() const;

    bool dirAutoUsesValue() const final;

    RefPtr<FileChooser> m_fileChooser;
    std::unique_ptr<FileIconLoader> m_fileIconLoader;

    Ref<FileList> m_fileList;
    RefPtr<DirectoryFileListCreator> m_directoryFileListCreator;
    RefPtr<Icon> m_icon;
    String m_displayString;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_INPUT_TYPE(FileInputType, Type::File)
