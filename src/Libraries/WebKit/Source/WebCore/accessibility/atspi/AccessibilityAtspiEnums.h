/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 16, 2025.
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

#if USE(ATSPI)
namespace WebCore {
namespace Atspi {

enum class Role {
    InvalidRole,
    AcceleratorLabel,
    Alert,
    Animation,
    Arrow,
    Calendar,
    Canvas,
    CheckBox,
    CheckMenuItem,
    ColorChooser,
    ColumnHeader,
    ComboBox,
    DateEditor,
    DesktopIcon,
    DesktopFrame,
    Dial,
    Dialog,
    DirectoryPane,
    DrawingArea,
    FileChooser,
    Filler,
    FocusTraversable,
    FontChooser,
    Frame,
    GlassPane,
    HtmlContainer,
    Icon,
    Image,
    InternalFrame,
    Label,
    LayeredPane,
    List,
    ListItem,
    Menu,
    MenuBar,
    MenuItem,
    OptionPane,
    PageTab,
    PageTabList,
    Panel,
    PasswordText,
    PopupMenu,
    ProgressBar,
    PushButton,
    RadioButton,
    RadioMenuItem,
    RootPane,
    RowHeader,
    ScrollBar,
    ScrollPane,
    Separator,
    Slider,
    SpinButton,
    SplitPane,
    StatusBar,
    Table,
    TableCell,
    TableColumnHeader,
    TableRowHeader,
    TearoffMenuItem,
    Terminal,
    Text,
    ToggleButton,
    ToolBar,
    ToolTip,
    Tree,
    TreeTable,
    Unknown,
    Viewport,
    Window,
    Extended,
    Header,
    Footer,
    Paragraph,
    Ruler,
    Application,
    Autocomplete,
    Editbar,
    Embedded,
    Entry,
    Chart,
    Caption,
    DocumentFrame,
    Heading,
    Page,
    Section,
    RedundantObject,
    Form,
    Link,
    InputMethodWindow,
    TableRow,
    TreeItem,
    DocumentSpreadsheet,
    DocumentPresentation,
    DocumentText,
    DocumentWeb,
    DocumentEmail,
    Comment,
    ListBox,
    Grouping,
    ImageMap,
    Notification,
    InfoBar,
    LevelBar,
    TitleBar,
    BlockQuote,
    Audio,
    Video,
    Definition,
    Article,
    Landmark,
    Log,
    Marquee,
    Math,
    Rating,
    Timer,
    Static,
    MathFraction,
    MathRoot,
    Subscript,
    Superscript,
    DescriptionList,
    DescriptionTerm,
    DescriptionValue,
    Footnote,
    ContentDeletion,
    ContentInsertion,
    Mark,
    Suggestion,
};

enum class State : uint64_t {
    InvalidState            = 1LLU << 0,
    Active                  = 1LLU << 1,
    Armed                   = 1LLU << 2,
    Busy                    = 1LLU << 3,
    Checked                 = 1LLU << 4,
    Collapsed               = 1LLU << 5,
    Defunct                 = 1LLU << 6,
    Editable                = 1LLU << 7,
    Enabled                 = 1LLU << 8,
    Expandable              = 1LLU << 9,
    Expanded                = 1LLU << 10,
    Focusable               = 1LLU << 11,
    Focused                 = 1LLU << 12,
    HasTooltip              = 1LLU << 13,
    Horizontal              = 1LLU << 14,
    Iconified               = 1LLU << 15,
    Modal                   = 1LLU << 16,
    MultiLine               = 1LLU << 17,
    Multiselectable         = 1LLU << 18,
    Opaque                  = 1LLU << 19,
    Pressed                 = 1LLU << 20,
    Resizable               = 1LLU << 21,
    Selectable              = 1LLU << 22,
    Selected                = 1LLU << 23,
    Sensitive               = 1LLU << 24,
    Showing                 = 1LLU << 25,
    SingleLine              = 1LLU << 26,
    Stale                   = 1LLU << 27,
    Transient               = 1LLU << 28,
    Vertical                = 1LLU << 29,
    Visible                 = 1LLU << 30,
    ManagesDescendants      = 1LLU << 31,
    Indeterminate           = 1LLU << 32,
    Required                = 1LLU << 33,
    Truncated               = 1LLU << 34,
    Animated                = 1LLU << 35,
    InvalidEntry            = 1LLU << 36,
    SupportsAutocompletion  = 1LLU << 37,
    SelectableText          = 1LLU << 38,
    IsDefault               = 1LLU << 39,
    Visited                 = 1LLU << 40,
    Checkable               = 1LLU << 41,
    HasPopup                = 1LLU << 42,
    ReadOnly                = 1LLU << 43,
};

enum class Relation {
    Null,
    LabelFor,
    LabelledBy,
    ControllerFor,
    ControlledBy,
    MemberOf,
    TooltipFor,
    NodeChildOf,
    NodeParentOf,
    ExtendedRelation,
    FlowsTo,
    FlowsFrom,
    SubwindowOf,
    Embeds,
    EmbeddedBy,
    PoupFor,
    ParentWindowOf,
    DescriptionFor,
    DescribedBy,
    Details,
    DetailsFor,
    ErrorMessage,
    ErrorFor,
};

enum class CoordinateType {
    ScreenCoordinates,
    WindowCoordinates,
    ParentCoordinates,
};

enum class ComponentLayer {
    InvalidLayer,
    BackgroundLayer,
    CanvasLayer,
    WidgetLayer,
    MdiLayer,
    PopupLayer,
    OverlayLayer,
    WindowLayer,
};

enum class ScrollType {
    TopLeft,
    BottomRight,
    TopEdge,
    BottomEdge,
    LeftEdge,
    RightEdge,
    Anywhere
};

enum class TextBoundaryType {
    CharBoundary,
    WordStartBoundary,
    WordEndBoundary,
    SentenceStartBoundary,
    SentenceEndBoundary,
    LineStartBoundary,
    LineEndBoundary
};

enum class TextGranularityType {
    CharGranularity,
    WordGranularity,
    SentenceGranularity,
    LineGranularity,
    ParagraphGranularity
};

enum class CollectionMatchType {
    MatchInvalid,
    MatchAll,
    MatchAny,
    MatchNone,
    MatchEmpty
};

enum class CollectionSortOrder {
    SortOrderInvalid,
    SortOrderCanonical,
    SortOrderFlow,
    SortOrderTab,
    SortOrderReverseCanonical,
    SortOrderReverseFlow,
    SortOrderReverseTab
};

} // namespace Atspi
} // namespace WebCore

#endif // USE(ATSPI)
