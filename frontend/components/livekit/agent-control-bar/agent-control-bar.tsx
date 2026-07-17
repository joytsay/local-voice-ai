'use client';

import { type HTMLAttributes, useCallback, useEffect, useMemo, useState } from 'react';
import { Track } from 'livekit-client';
import { useChat, useRemoteParticipants } from '@livekit/components-react';
import {
  ChatTextIcon,
  PhoneDisconnectIcon,
  SpinnerGapIcon,
  UploadSimpleIcon,
} from '@phosphor-icons/react/dist/ssr';
import { TrackToggle } from '@/components/livekit/agent-control-bar/track-toggle';
import { Button } from '@/components/livekit/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/livekit/select';
import { Toggle } from '@/components/livekit/toggle';
import { cn } from '@/lib/utils';
import { ChatInput } from './chat-input';
import { UseInputControlsProps, useInputControls } from './hooks/use-input-controls';
import { usePublishPermissions } from './hooks/use-publish-permissions';
import { TrackSelector } from './track-selector';

export interface ControlBarControls {
  leave?: boolean;
  camera?: boolean;
  microphone?: boolean;
  screenShare?: boolean;
  chat?: boolean;
}

export interface AgentControlBarProps extends UseInputControlsProps {
  controls?: ControlBarControls;
  isConnected?: boolean;
  onChatOpenChange?: (open: boolean) => void;
  onDeviceError?: (error: { source: Track.Source; error: Error }) => void;
}

interface ModelCatalog {
  stt: {
    current: string;
    options: string[];
    managed: boolean;
  };
  tts: {
    current: string;
    options: string[];
    managed: boolean;
  };
}

/**
 * A control bar specifically designed for voice assistant interfaces
 */
export function AgentControlBar({
  controls,
  saveUserChoices = true,
  className,
  isConnected = false,
  onDisconnect,
  onDeviceError,
  onChatOpenChange,
  ...props
}: AgentControlBarProps & HTMLAttributes<HTMLDivElement>) {
  const { send } = useChat();
  const participants = useRemoteParticipants();
  const [chatOpen, setChatOpen] = useState(false);
  const [modelCatalog, setModelCatalog] = useState<ModelCatalog | null>(null);
  const [selectedSttModel, setSelectedSttModel] = useState('');
  const [selectedTtsModel, setSelectedTtsModel] = useState('');
  const [modelLoadState, setModelLoadState] = useState<'idle' | 'loading' | 'error'>('idle');
  const publishPermissions = usePublishPermissions();
  const {
    micTrackRef,
    cameraToggle,
    microphoneToggle,
    screenShareToggle,
    handleAudioDeviceChange,
    handleVideoDeviceChange,
    handleMicrophoneDeviceSelectError,
    handleCameraDeviceSelectError,
  } = useInputControls({ onDeviceError, saveUserChoices });

  const handleSendMessage = async (message: string) => {
    await send(message);
  };

  const handleToggleTranscript = useCallback(
    (open: boolean) => {
      setChatOpen(open);
      onChatOpenChange?.(open);
    },
    [onChatOpenChange, setChatOpen]
  );

  const visibleControls = {
    leave: controls?.leave ?? true,
    microphone: controls?.microphone ?? publishPermissions.microphone,
    screenShare: controls?.screenShare ?? publishPermissions.screenShare,
    camera: controls?.camera ?? publishPermissions.camera,
    chat: controls?.chat ?? publishPermissions.data,
  };

  const isAgentAvailable = participants.some((p) => p.isAgent);
  const modelControlsDisabled = modelLoadState === 'loading' || !modelCatalog;
  const hasModelChanges = useMemo(() => {
    if (!modelCatalog) {
      return false;
    }
    return (
      selectedSttModel !== modelCatalog.stt.current || selectedTtsModel !== modelCatalog.tts.current
    );
  }, [modelCatalog, selectedSttModel, selectedTtsModel]);

  useEffect(() => {
    let cancelled = false;

    async function loadModelCatalog() {
      try {
        const response = await fetch('/api/models', { cache: 'no-store' });
        if (!response.ok) {
          throw new Error(`Model catalog request failed: ${response.status}`);
        }
        const data = (await response.json()) as ModelCatalog;
        if (cancelled) {
          return;
        }
        setModelCatalog(data);
        setSelectedSttModel(data.stt.current);
        setSelectedTtsModel(data.tts.current);
      } catch {
        if (!cancelled) {
          setModelLoadState('error');
        }
      }
    }

    loadModelCatalog();
    return () => {
      cancelled = true;
    };
  }, []);

  const handleReloadModels = useCallback(async () => {
    if (!modelCatalog || !hasModelChanges) {
      return;
    }

    setModelLoadState('loading');
    try {
      const response = await fetch('/api/models/reload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sttModel: selectedSttModel !== modelCatalog.stt.current ? selectedSttModel : undefined,
          ttsModel: selectedTtsModel !== modelCatalog.tts.current ? selectedTtsModel : undefined,
        }),
      });
      if (!response.ok) {
        throw new Error(`Model reload failed: ${response.status}`);
      }
      const reloaded = (await response.json()) as { sttModel: string; ttsModel: string };
      setModelCatalog((current) =>
        current
          ? {
              stt: { ...current.stt, current: reloaded.sttModel },
              tts: { ...current.tts, current: reloaded.ttsModel },
            }
          : current
      );
      setSelectedSttModel(reloaded.sttModel);
      setSelectedTtsModel(reloaded.ttsModel);
      setModelLoadState('idle');
    } catch {
      setModelLoadState('error');
    }
  }, [hasModelChanges, modelCatalog, selectedSttModel, selectedTtsModel]);

  return (
    <div
      aria-label="Voice assistant controls"
      className={cn(
        'bg-background border-input/50 dark:border-muted flex flex-col rounded-[31px] border p-3 drop-shadow-md/3',
        className
      )}
      {...props}
    >
      {/* Chat Input */}
      {visibleControls.chat && (
        <ChatInput
          chatOpen={chatOpen}
          isAgentAvailable={isAgentAvailable}
          onSend={handleSendMessage}
        />
      )}

      <div className="mb-2 grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] gap-1">
        <Select
          value={selectedSttModel}
          disabled={modelControlsDisabled || modelCatalog?.stt.managed === false}
          onValueChange={setSelectedSttModel}
        >
          <SelectTrigger aria-label="STT model" className="h-9 w-full min-w-0 rounded-[18px]">
            <SelectValue placeholder="STT" />
          </SelectTrigger>
          <SelectContent>
            {modelCatalog?.stt.options.map((model) => (
              <SelectItem key={model} value={model}>
                {model}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Select
          value={selectedTtsModel}
          disabled={modelControlsDisabled || modelCatalog?.tts.managed === false}
          onValueChange={setSelectedTtsModel}
        >
          <SelectTrigger aria-label="TTS model" className="h-9 w-full min-w-0 rounded-[18px]">
            <SelectValue placeholder="TTS" />
          </SelectTrigger>
          <SelectContent>
            {modelCatalog?.tts.options.map((model) => (
              <SelectItem key={model} value={model}>
                {model}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <Button
          type="button"
          variant={modelLoadState === 'error' ? 'destructive' : 'secondary'}
          size="icon"
          aria-label="Load selected speech models"
          disabled={modelControlsDisabled || !hasModelChanges}
          onClick={handleReloadModels}
          title={modelLoadState === 'error' ? 'Model reload failed' : 'Load selected speech models'}
        >
          {modelLoadState === 'loading' ? (
            <SpinnerGapIcon className="animate-spin" weight="bold" />
          ) : (
            <UploadSimpleIcon weight="bold" />
          )}
        </Button>
      </div>

      <div className="flex gap-1">
        <div className="flex grow gap-1">
          {/* Toggle Microphone */}
          {visibleControls.microphone && (
            <TrackSelector
              kind="audioinput"
              aria-label="Toggle microphone"
              source={Track.Source.Microphone}
              pressed={microphoneToggle.enabled}
              disabled={microphoneToggle.pending}
              audioTrackRef={micTrackRef}
              onPressedChange={microphoneToggle.toggle}
              onMediaDeviceError={handleMicrophoneDeviceSelectError}
              onActiveDeviceChange={handleAudioDeviceChange}
            />
          )}

          {/* Toggle Camera */}
          {visibleControls.camera && (
            <TrackSelector
              kind="videoinput"
              aria-label="Toggle camera"
              source={Track.Source.Camera}
              pressed={cameraToggle.enabled}
              pending={cameraToggle.pending}
              disabled={cameraToggle.pending}
              onPressedChange={cameraToggle.toggle}
              onMediaDeviceError={handleCameraDeviceSelectError}
              onActiveDeviceChange={handleVideoDeviceChange}
            />
          )}

          {/* Toggle Screen Share */}
          {visibleControls.screenShare && (
            <TrackToggle
              size="icon"
              variant="secondary"
              aria-label="Toggle screen share"
              source={Track.Source.ScreenShare}
              pressed={screenShareToggle.enabled}
              disabled={screenShareToggle.pending}
              onPressedChange={screenShareToggle.toggle}
            />
          )}

          {/* Toggle Transcript */}
          <Toggle
            size="icon"
            variant="secondary"
            aria-label="Toggle transcript"
            pressed={chatOpen}
            onPressedChange={handleToggleTranscript}
          >
            <ChatTextIcon weight="bold" />
          </Toggle>
        </div>

        {/* Disconnect */}
        {visibleControls.leave && (
          <Button
            variant="destructive"
            onClick={onDisconnect}
            disabled={!isConnected}
            className="font-mono"
          >
            <PhoneDisconnectIcon weight="bold" />
            <span className="hidden md:inline">END CALL</span>
            <span className="inline md:hidden">END</span>
          </Button>
        )}
      </div>
    </div>
  );
}
