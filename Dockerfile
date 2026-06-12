EXPOSE 7860
CMD ["gunicorn", "--workers", "2", "--timeout", "180", "-b", "0.0.0.0:7860", "app:app"]
