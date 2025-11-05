# Heroku Deployment Guide

This guide will help you deploy the Podcast Agent application to Heroku.

## Prerequisites

1. **Heroku Account**: Sign up at [heroku.com](https://heroku.com)
2. **Heroku CLI**: Install from [devcenter.heroku.com/articles/heroku-cli](https://devcenter.heroku.com/articles/heroku-cli)
3. **Git**: Ensure your code is in a git repository
4. **API Keys**: Have your API keys ready:
   - Anthropic API Key (required)
   - Llama API Key (required)
   - At least one search API key (Tavily, Serper, or Brave)

## Deployment Steps

### 1. Login to Heroku

```bash
heroku login
```

### 2. Create a New Heroku App

```bash
heroku create your-app-name
```

Or let Heroku generate a random name:

```bash
heroku create
```

### 3. Set Environment Variables

Set all required API keys as Heroku config vars:

```bash
# Required LLM API Keys
heroku config:set ANTHROPIC_API_KEY=your_anthropic_key
heroku config:set LLAMA_API_KEY=your_llama_key

# Required: At least one search tool API key
heroku config:set TAVILY_API_KEY=your_tavily_key
# OR
heroku config:set SERPER_API_KEY=your_serper_key
# OR
heroku config:set BRAVE_API_KEY=your_brave_key

# Optional: LangSmith tracing (for debugging)
heroku config:set LANGSMITH_TRACING=false
heroku config:set LANGSMITH_API_KEY=your_langsmith_key
heroku config:set LANGSMITH_PROJECT=your_project_name
```

You can also set these in the Heroku Dashboard under Settings > Config Vars.

### 4. Deploy to Heroku

```bash
git push heroku main
```

If you're on a different branch:

```bash
git push heroku your-branch:main
```

### 5. Verify Deployment

Check if the app is running:

```bash
heroku logs --tail
```

Open your app in a browser:

```bash
heroku open
```

## Scaling

By default, Heroku will run one web dyno. To scale:

```bash
# Scale up
heroku ps:scale web=1

# Scale down (to save dyno hours)
heroku ps:scale web=0
```

## Monitoring

### View Logs

```bash
# Real-time logs
heroku logs --tail

# Last 1000 lines
heroku logs -n 1000

# Filter by source
heroku logs --source app
```

### Check Dyno Status

```bash
heroku ps
```

### Restart the App

```bash
heroku restart
```

## Troubleshooting

### Build Fails

1. Check that `requirements.txt` is in the root directory
2. Verify `runtime.txt` specifies a supported Python version
3. Check build logs: `heroku logs --tail`

### Application Errors

1. Check application logs: `heroku logs --tail`
2. Verify all environment variables are set: `heroku config`
3. Test locally with the same environment variables

### Port Binding Issues

The app should automatically bind to the `$PORT` environment variable provided by Heroku. This is configured in the `Procfile`.

### Memory Issues

If the app runs out of memory:

```bash
# Upgrade to a larger dyno (requires paid plan)
heroku ps:scale web=1:standard-1x
```

## Updating the App

After making changes:

```bash
git add .
git commit -m "Your commit message"
git push heroku main
```

## Cost Considerations

- **Free Tier**: 550-1000 free dyno hours per month (requires credit card verification)
- **Hobby Tier**: $7/month per dyno (never sleeps)
- **Standard Tier**: Starting at $25/month (better performance)

Note: LLM API usage (Anthropic, Llama) will be billed separately by those providers.

## Additional Resources

- [Heroku Python Documentation](https://devcenter.heroku.com/categories/python-support)
- [Heroku CLI Commands](https://devcenter.heroku.com/articles/heroku-cli-commands)
- [Environment Variables](https://devcenter.heroku.com/articles/config-vars)

## Security Notes

1. **Never commit** API keys or `.env` files to git
2. Always use `heroku config:set` to manage secrets
3. Regularly rotate your API keys
4. Enable Heroku's automatic security updates

## Support

For issues specific to this application, check the main README.md or open an issue in the repository.

For Heroku-specific issues, visit [help.heroku.com](https://help.heroku.com).
