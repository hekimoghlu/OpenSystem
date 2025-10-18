; Test methods of Edit module of the PDB

; This tests some of the general cases.
; See elsewhere for other cases e.g. edit-multi-layer.scm

; TODO the "Paste In Place" methods are not in the PDB?

; TODO the "Paste As" distinctions between Layer and Floating Sel
; are not in the PDB


(script-fu-use-v3)


; setup
; Load test image that already has drawable
(define testImage (testing:load-test-image-basic-v3))

; get all the root layers
; testImage has exactly one root layer.
(define testLayers (gimp-image-get-layers testImage))
;testLayers is-a vector

(define testLayer (vector-ref testLayers 0))

; !!! Testing requires initial condition: clipboard is empty.
; But there is no programmatic way to empty the clipboard,
; or even to determine if the clipboard is empty.
; So these tests might not pass when you run this test file
; in the wrong order.




(test! "named-copy")

(define testBuffer (gimp-edit-named-copy
                              (make-vector 1 testLayer)
                              "testBufferName"))
; There is one named buffer
(assert `(= (length (gimp-buffers-get-name-list "")) 1))



(test! "paste")

; ordinary paste is to a drawable

; paste always returns a layer, even when clipboard empty
; FIXME: this is undocumented in the PDB,
; where it seems to imply that a layer is always returned.
(assert-error `(gimp-edit-paste
                   ,testLayer
                   TRUE) ; paste-into
              "Procedure execution of gimp-edit-paste failed" )

; paste-as-new-image returns NULL image when clipboard empty
; paste-as-new is deprecated
(assert `(= (gimp-edit-paste-as-new-image)
            -1))  ; the NULL ID
; named-paste-as-new-image returns a new image
(assert `(gimp-image-id-is-valid (gimp-edit-named-paste-as-new-image "testBufferName")))

(test! "copy")

; copy when:
;  - no selection
;  - image has one drawable
;  - one drawable is passed
; returns true and clip has one drawable
(assert `(gimp-edit-copy ,testLayers))

; paste when clipboard is not empty returns a vector of length one
(assert `(= (vector-length (gimp-edit-paste
                   ,testLayer
                   TRUE)) ; paste-into
            1))
; get reference to pasted layer
(define testPastedLayer (vector-ref (gimp-image-get-layers testImage)
                                       0))

; !!! this is not what happens in the GUI, the pasted layer is NOT floating
; The pasted layer is floating
(assert `(gimp-layer-is-floating-sel ,testPastedLayer))



(test! "copy-visible")

; copy-visible takes only an image
; it puts one drawable on clip
(assert `(gimp-edit-copy-visible ,testImage))




(test! "named-paste")

; There is one named buffer
(assert `(= (length (gimp-buffers-get-name-list "")) 1))

; named-paste returns just the floating sel
(assert `(gimp-edit-named-paste
                   ,testLayer
                   "testBufferName"
                   #f)) ; paste-into


(test! "paste  into")
; paste when clipboard is not empty returns a vector of length one
(assert `(= (vector-length (gimp-edit-paste ,testLayer TRUE)) ; paste-into
            1))

; The first pasted floating layer was anchored (merged into) first layer
; The ID of the first floating sel is now invalid !
(assert `(not (gimp-item-id-is-valid ,testPastedLayer)))
; The first floating sel MUST NOT BE USED AGAIN !
(assert-error `(gimp-layer-is-floating-sel ,testPastedLayer)
              "Invalid value for argument 0")


; There are now two layers
(assert `(= (vector-length (gimp-image-get-layers ,testImage)) 2))

(define testPastedLayer2 (vector-ref (gimp-image-get-layers testImage)
                                       0))
; the new layer is now floating.
(assert `(gimp-layer-is-floating-sel ,testPastedLayer2))




(test! "paste into empty image")

; paste specifying no destination layer is an error.
; You cannot paste into an empty image having no layers
; because you must pass a destination layer.
; !!! This is different from the GUI
(assert-error `(gimp-edit-paste -1 TRUE)
              "Invalid value for argument 0")




; TODO test cross image paste, of different modes


; copy returns false when is a selection
; but it doesn't intersect any chosen layers
; When one layer is offset from another and bounds of layer
; do not intersect bounds of selection.
; TODO

; cleanup
; delete buffers, other tests may expect no buffers
(gimp-buffer-delete "testBufferName")

; for debugging individual test file
(testing:show testImage)


(script-fu-use-v2)
