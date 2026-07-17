'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { SpinnerGapIcon, UploadSimpleIcon } from '@phosphor-icons/react/dist/ssr';
import { Button } from '@/components/livekit/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/livekit/select';
import { cn } from '@/lib/utils';

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

export function ModelReloadControls({ className }: { className?: string }) {
  const [modelCatalog, setModelCatalog] = useState<ModelCatalog | null>(null);
  const [selectedSttModel, setSelectedSttModel] = useState('');
  const [selectedTtsModel, setSelectedTtsModel] = useState('');
  const [modelLoadState, setModelLoadState] = useState<'idle' | 'loading' | 'error'>('idle');

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
        setModelLoadState('idle');
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
      className={cn('grid w-full grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] gap-2', className)}
    >
      <Select
        value={selectedSttModel}
        disabled={modelControlsDisabled || modelCatalog?.stt.managed === false}
        onValueChange={setSelectedSttModel}
      >
        <SelectTrigger aria-label="STT model" className="h-9 w-full min-w-0 rounded-[18px]">
          <SelectValue placeholder="STT model" />
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
          <SelectValue placeholder="TTS model" />
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
  );
}
