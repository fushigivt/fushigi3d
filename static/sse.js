// HTMX SSE Extension
// Enables Server-Sent Events for HTMX
// Original: https://unpkg.com/htmx.org/dist/ext/sse.js

(function(){
    var api;

    htmx.defineExtension('sse', {
        init: function(apiRef) {
            api = apiRef;
        },

        onEvent: function(name, evt) {
            var parent = evt.target;
            if (name === "htmx:beforeCleanupElement") {
                var internalData = api.getInternalData(parent);
                if (internalData.sseEventSource) {
                    internalData.sseEventSource.close();
                }
                return;
            }

            if (name !== "htmx:afterProcessNode") {
                return;
            }

            var sseConnect = api.getAttributeValue(parent, "sse-connect");
            if (sseConnect) {
                processSSEConnect(parent, sseConnect);
            }
        }
    });

    function processSSEConnect(elt, url) {
        var internalData = api.getInternalData(elt);
        var source = new EventSource(url);
        internalData.sseEventSource = source;

        source.onerror = function(e) {
            api.triggerErrorEvent(elt, "htmx:sseError", {error: e, source: source});
        };

        // Process sse-swap attributes
        var sseSwapAttr = api.getAttributeValue(elt, "sse-swap");
        if (sseSwapAttr) {
            var sseSwapSpec = sseSwapAttr.split(/[\s,]+/);
            sseSwapSpec.forEach(function(eventName) {
                source.addEventListener(eventName, function(event) {
                    var settleInfo = api.makeSettleInfo(elt);
                    var swap = api.getAttributeValue(elt, "hx-swap") || "innerHTML";
                    api.swap(elt, event.data, {
                        swapStyle: swap,
                        settleInfo: settleInfo
                    });
                    api.settleImmediately(settleInfo.tasks);
                });
            });
        }

        // Forward SSE events as HTMX triggers
        source.onmessage = function(event) {
            var data;
            try {
                data = JSON.parse(event.data);
            } catch (e) {
                data = event.data;
            }
            htmx.trigger(elt, "sse:message", {data: data});
        };

        // Listen for typed events
        var children = elt.querySelectorAll("[hx-trigger*='sse:']");
        children.forEach(function(child) {
            var trigger = api.getAttributeValue(child, "hx-trigger");
            var match = trigger.match(/sse:(\w+)/);
            if (match) {
                var eventName = match[1];
                source.addEventListener(eventName, function(event) {
                    htmx.trigger(child, "sse:" + eventName, {data: event.data});
                });
            }
        });

        // Also check parent element
        var trigger = api.getAttributeValue(elt, "hx-trigger");
        if (trigger) {
            var match = trigger.match(/sse:(\w+)/);
            if (match) {
                var eventName = match[1];
                source.addEventListener(eventName, function(event) {
                    htmx.trigger(elt, "sse:" + eventName, {data: event.data});
                });
            }
        }
    }
})();
